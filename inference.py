import os
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
# from torch.optim import AdamW
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import argparse
import weight_evaluation


# from transformers import AdamW
import time
import datetime
import random
from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
import sys
from dateutil import tz

from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset
from weight_evaluation import sample_weighted_f1_score, sample_weighted_precision, sample_weighted_recall, estimate_threshold


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.encoding = "UTF-8"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


class Evaluate:
    def __init__(
        self,
        model_name: str,
        logdir: str,
        test_df,
        view1_name: str,
        view2_name: str,
        max_length: int,
        batch_size: int,
        num_classes: int,
        threshold: float,
        target: str,
        model1: str,
        model2: str,
        weight_file_path: str,
        recall_thresh:float,
        ensemble_mode:str
    ):
        # set your own timezone
        tzone = tz.gettz("America/Edmonton")
        self.timestamp = (
            datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")
        )

        self.model_name = model_name
        self.view1 = view1_name
        self.view2 = view2_name

        # writer to log information:
        self.logdir = logdir
        self.logger = Logger(os.path.join(
            self.logdir, self.timestamp + ".log"))
        sys.stdout = self.logger
        sys.stderr = self.logger
        
        # assign all of the various data sets that we need
        self.test_df = test_df
        self.weight_file_path = weight_file_path
       
        # Set training parameters of each separate model:
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.target = target
        self.threshold = threshold

        # threshold for recall level
        self.recall_thresh = recall_thresh
        self.ensemble_mode = ensemble_mode

        # Load datasets
        self.view1_test_dataloader = self.load_dataset(
            section=self.view1,
            df=self.test_df.reset_index(drop=True),
            labeled=False,
            shuffle=False
        )

        self.view2_test_dataloader = self.load_dataset(
            section=self.view2,
            df=self.test_df.reset_index(drop=True),
            labeled=False,
            shuffle=False
        )

        # Both views are initialized from the same model
        self.model1 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        ).cuda()


        self.model2 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False, # Whether the model returns all hidden-states.
        ).cuda()

        checkpoint = torch.load(model1)
        self.model1.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model1.cuda()

        checkpoint = torch.load(model2)
        self.model2.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model2.cuda()

    def load_dataset(self, section, df, labeled, shuffle=True):
        # load data into dataloader + tokenize
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name)
        if labeled:
            dataset = RadiologyLabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                target=self.target,
                view_name=section
            )
        else:
            dataset = RadiologyUnlabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                view_name=section
            )

        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8, pin_memory=True,
        )
        return dataloader
    
    def thresh_pred_prob(self,probabilities):
        # probabilities are only for event probabilities
        
        pred = []
        for i in range(len(probabilities)):
            if probabilities[i] >= self.threshold:
                pred.append(1)
            else: 
                pred.append(0)
        return pred
    
    def ensemble_eval(self):
        # store the prob, preds and labels
        view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
            model=self.model1, dataloader=self.view1_test_dataloader)
        view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
            model=self.model2, dataloader=self.view2_test_dataloader)

        # first check if the view1 and view2 labels are the same:
        if view1_file_names == view2_file_names:
            print("The files names are the same")
        else:
            print("Check code")

        # combine the probabilities together
        # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
        probabilities = np.add(view1_probs, view2_probs)
        avg_probs = np.divide(probabilities, 2)
        pred_labels = self.thresh_pred_prob(avg_probs[:,1].flatten().tolist())
        model_pred_df = pd.DataFrame({"file_names":view1_file_names, 
                                      "labels":view1_labels, 
                                      "predicted": pred_labels, 
                                      "probabilities":avg_probs[:,1].flatten().tolist(),
                                      "findings_prob": view1_probs[:,1].flatten().tolist(),
                                      "impression_prob":view2_probs[:,1].flatten().tolist(),
                                      })
        # model_pred_df.to_csv(self.target+'_ensemble_results.csv')

        print("Results for ensemble")

        # accuracy:
        ensemble_f1 = weight_evaluation.sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print("Sample F1 : {0: .6f} ".format(ensemble_f1))
        ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(view1_labels)
        print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
        ensemble_recall =  weight_evaluation.sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print("Sample Recall : {0: .6f} ".format(ensemble_recall))

        weight_evaluation.estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=avg_probs[:,1].flatten().tolist(), target=self.target, recall_thresh=self.recall_thresh, visualize=False)

        return ensemble_f1, view1_f1, view2_f1
    
    def eval(self, model, dataloader):
         # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        probs = np.zeros((0, self.num_classes))
        probs_1 = []
        preds = []
        labels = []
        pids = []
        accessions = []

        model.eval()
        for batch in enumerate(dataloader):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()
            b_pid = batch["pid"].cuda()
            b_accession = batch['accession'].cuda()

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits

            # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
            probabilities = (softmax(logits).detach().cpu().numpy())
            event_probs = [p[1] for p in probabilities]
            probs_1 += event_probs
            predictions = (
                np.argmax(logits.detach().cpu().numpy(),
                          axis=1).flatten().tolist()
            )
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            probs = np.concatenate((probs, probabilities), axis=0)
            preds += predictions
            labels += label_ids
            b_pid = batch["pid"].cuda()
            b_accession = batch['accession'].cuda()
        
        # perform thresholding to adjust the preds
        preds = self.thresh_pred_prob(probs[:,1].flatten().tolist())
        model_pred_df = pd.DataFrame({"anon_accession": accessions, "anon_PID":pids, "predicted": preds, "event_probability": probs_1})
        # model_pred_df.to_csv('model_pred_df.csv')

        sample_f1 =  weight_evaluation.sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print("Sample F1 : {0: .6f} ".format(sample_f1))
        accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
        print("Accuracy: ", accuracy)
        sample_recall =  weight_evaluation.sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print('Recall: ', sample_recall)
        precision = (sample_f1 * sample_recall) / (2 * sample_recall - sample_f1)
        print('Precision: ', precision)

        weight_evaluation.estimate_threshold(self.weight_file_path,model_pred_df=model_pred_df, model_event_prob=probs[:,1].flatten().tolist(), target=self.target, recall_thresh=self.recall_thresh)
        return sample_f1, probs, labels, pids, accessions

    
    def inference(self, model, dataloader):
        """
        Make predictions on unlabeled data, returns probabilities and identifiers.
        """

        softmax = nn.Softmax(dim=-1)

        probs_1 = []
        preds = []
        file_names = []

        model.eval()
        for batch in tqdm(dataloader, desc="Running inference"):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()

            # keep the whole batch of file names, not only the first one
            b_file_names = list(batch["file"])

            with torch.no_grad():
                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )

            logits = result.logits
            probabilities = softmax(logits).detach().cpu().numpy()

            event_probs = probabilities[:, 1].tolist()
            batch_preds = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten().tolist()

            probs_1.extend(event_probs)
            preds.extend(batch_preds)
            file_names.extend(b_file_names)

        # threshold on event probabilities
        preds = self.thresh_pred_prob(probs_1)

        print("len(file_names):", len(file_names))
        print("len(probs_1):", len(probs_1))
        print("len(preds):", len(preds))

        assert len(file_names) == len(probs_1) == len(preds), \
            f"Length mismatch: file_names={len(file_names)}, probs_1={len(probs_1)}, preds={len(preds)}"

        return probs_1, file_names


    def ensemble_inference(self):
        # store the prob, preds and labels
        view1_probs, view1_file_names = self.inference(
            model=self.model1, dataloader=self.view1_test_dataloader)
        view2_probs, view2_file_names = self.inference(
            model=self.model2, dataloader=self.view2_test_dataloader)

        # first check if the view1 and view2 labels are the same:
        if view1_file_names == view2_file_names:
            print("The pids for two models are the same")
        else:
            print("Check code")

        # combine the probabilities together
        # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
        print('view1_prob: ', view1_probs)
        print('view2_prob: ', view2_probs)
        if self.ensemble_mode == 'average':
            probabilities = np.add(view1_probs, view2_probs)
            probabilities = np.divide(probabilities, 2)
            
        else:
            # maximize
            probabilities = np.maximum(view1_probs, view2_probs)
            pred_labels = self.thresh_pred_prob(probabilities)
        print('ensembled_probs: ', probabilities)
        pred_labels = self.thresh_pred_prob(probabilities)
        model_pred_df = pd.DataFrame({"file_names":view1_file_names, 
                                      "predicted": pred_labels, 
                                      "probabilities":probabilities,
                                      "findings_prob": view1_probs,
                                      "impression_prob":view2_probs,
                                      })

        return model_pred_df


if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--max-length",
        type=int,
        help="The max number of tokens per sequence",
        default=512,
    )
    argparser.add_argument("--random-seed", type=int, default=0)
    argparser.add_argument('--labeled', action='store_true')
    

    # Model Settings:
    argparser.add_argument("--model-name", type=str,
                           default="emilyalsentzer/Bio_ClinicalBERT")
    argparser.add_argument("--model1", type=str, 
                           default='logs/ensemble_both_rounds_0/findings_epoch6_met_label_x.pt', 
                           help='The path to view1 checkpoint')
    argparser.add_argument("--model2", type=str, 
                           default='logs/ensemble_both_rounds_0/impressions_epoch6_met_label_x.pt',
                           help='The path to view2 checkpoint')
    argparser.add_argument("--view1-name", type=str, default="findings")
    argparser.add_argument("--view2-name", type=str, default="impressions")

    # Training
    argparser.add_argument("--batch-size", type=int, default=16)
    argparser.add_argument("--num_classes", type=int, default=2)
    argparser.add_argument("--target", type=str, default="met_label_x")
    argparser.add_argument("--threshold", type=float, default=0.4)
    argparser.add_argument("--recall-thresh", type=float, default=0.9)
    argparser.add_argument('--ensemble-mode', type=str, default='max', help='max or average')

    # Path to tensorboard and csv of results:
    argparser.add_argument(
        "--logdir",
        type=str,
        default="logs/eval_test",
        help="Path to save results to",
    )
    argparser.add_argument("--test-data-pkl", type=str, default='Data/post_diag_reports_processed.pkl', help='Data pkl path for inference/evaluation')
    argparser.add_argument("--weight-path", type=str, default='', help='For weighted evaluation')

    args = argparser.parse_args()

    # Set device:
    device = torch.device("cuda")

    torch.cuda.empty_cache()

    test_df = pd.read_pickle(args.test_data_pkl)
    print(test_df.shape)


    # Set the seed value all over the place to make this reproducible.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    os.makedirs(args.logdir, exist_ok=True)
    
    method = Evaluate(
        model_name=args.model_name,
        view1_name=args.view1_name,
        view2_name=args.view2_name,
        logdir=args.logdir,
        test_df=test_df,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        target=args.target,
        threshold=args.threshold,
        model1=args.model1,
        model2=args.model2,
        weight_file_path=args.weight_path,
        recall_thresh=args.recall_thresh,
        ensemble_mode=args.ensemble_mode
    )
    if args.labeled:
        method.ensemble_eval()
    else:
        predicted_df = method.ensemble_inference()
        # predicted_df.to_csv(args.target+'_acr_ir_cohort_max_260224.csv')
        predicted_df.to_csv(args.target+'_acr_ir_cohort_average04986_260417.csv')
        # predicted_df.to_csv(args.target+'_predictions_avg_ext_threshold05_0419.csv')
        # acr = pd.read_csv('Data/present_acr_ir_nonmelanoma_prediag_cns_excluded.csv')
        # predicted_df = predicted_df.rename(columns={'file_names': 'File Name'})
        # predicted_df = predicted_df.merge(acr[['File Name', 'New File Name', 'Study Date', 'ID']], on='File Name', how='left')
        # predicted_df.to_csv(args.target+'_predictions_avg_method3_threshold05_0422.csv')
