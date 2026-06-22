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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
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
import math
import sys
from dateutil import tz

from data_processing import organize_labeled_data, organize_unlabeled_data
from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset
from weight_evaluation_weighted import estimate_weighted_f1_score, sample_weighted_f1_score, sample_weighted_precision
import weight_evaluation
from torch.utils.tensorboard import SummaryWriter


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


class BaseModel:
    def __init__(
        self,
        model_name: str,
        logdir: str,
        train_df,
        test_df,
        val_df,
        view1_name: str,
        view2_name: str,
        max_length: int,
        batch_size: int,
        num_classes: int,
        learning_rate: float,
        threshold: float,
        num_epochs: int,
        target: str,
        use_weight: bool,
        validate: bool,
        weight_file_path:str,
        early_stop:bool,
        args:str
    ):
        tzone = tz.gettz("America/Edmonton")
        self.timestamp = (
            datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")
        )

        self.model_name = model_name
        self.view1 = view1_name
        self.view2 = view2_name

        # writer to log information:
        self.logdir = logdir
        self.writer = SummaryWriter(self.logdir)
        self.logger = Logger(os.path.join(
            self.logdir, self.timestamp + ".log"))
        sys.stdout = self.logger
        sys.stderr = self.logger
        print(args)
        print()
        self.results_df = pd.DataFrame(columns=["Ensemble accuracy",
                                                "View1 accuracy",
                                                "View2 accuracy",
                                                'Val/Test'])

        # assign all of the various data sets that we need
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.validate = validate
        self.weight_file_path = weight_file_path
        self.early_stop = early_stop

        # Set training parameters of each separate model:
        self.global_epoch = 0
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.target = target
        self.threshold = threshold
        self.use_weight = use_weight

        # Load datasets
        self.view1_test_dataloader = self.load_dataset(
            section=self.view1,
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view2_test_dataloader = self.load_dataset(
            section=self.view2,
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )


        self.view1_val_dataloader = self.load_dataset(
            section=self.view1,
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        self.view2_val_dataloader = self.load_dataset(
            section=self.view2,
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        # Both views are initialized from the same model
        self.init_model1 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        ).cuda()

        self.init_model2 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,  # The number of output labels
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
        ).cuda()

        # Set optimizers for each view:
        self.view1_optimizer = AdamW(
            self.init_model1.parameters(), lr=self.learning_rate)
        self.view2_optimizer = AdamW(
            self.init_model2.parameters(), lr=self.learning_rate)

    def load_dataset(self, section, df, labeled=True, other_section="", shuffle=True):
        # load data into dataloader + tokenize
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name)
        if labeled:
            dataset = RadiologyLabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                target=self.target,
                view_name=section,
                use_weight=self.use_weight
            )
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
            )
        else:
            dataset = RadiologyUnlabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                view_name=section,
                other_view_name=other_section,
            )

            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=False
            )
        return dataloader

    def save_checkpoint(self, model, optimizer, section):
        filename = section + "_" + "epoch" + str(self.global_epoch) + "_" + self.target + ".pt"
        torch.save(
            {
                "epoch": self.global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(self.logdir, filename),
        )

    def resume_from_checkpoint(self, section):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        filename = section + "_" + "epoch" + str(self.global_epoch) + "_" + self.target + ".pt"

        checkpoint = torch.load(os.path.join(self.logdir, filename))

        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    def train(self, model, val_dataloader, test_dataloader, df, section, optimizer):

        dataloader = self.load_dataset(section, df.reset_index(drop=True))

        last_epoch_f1 = 0

        for epoch_i in range(self.num_epochs):
            sum_loss = 0
            sum_correct = 0
            model.train()
            # For each batch of training data...
            for i, batch in enumerate(dataloader):
                # Unpack this training batch from our dataloader.
                b_input_ids = batch["ids"].cuda()
                b_input_mask = batch["mask"].cuda()
                b_labels = batch["target"].cuda()
                if self.use_weight:
                    b_weight = batch["weight"].cuda()

                model.zero_grad()

                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )
                # _, predicted = torch.max(result.data, 1)

                if self.use_weight: 
                    # to use sample weights, discard loss returned by Bert and use the logits to calculate weighted new loss
                    loss_fn = nn.CrossEntropyLoss(reduction='none')
                    logits = result.logits
                    loss = loss_fn(logits, b_labels)
                    loss = loss * b_weight
                    loss = torch.mean(loss)
                    self.writer.add_scalar('Weighted_loss_'+section, loss.item(), len(dataloader)*epoch_i+i)
                else:
                    loss = result.loss
                    self.writer.add_scalar('Loss_'+section, loss.item(), len(dataloader)*epoch_i+i)


                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # sum_loss += loss.item()
                # sum_correct = (predicted == b_labels).sum().item()
                # if i == len(dataloader) - 1:
                #     print('Epoch {} Loss {}'.format(epoch_i, sum_loss / len(dataloader)))
                #     print()

            self.global_epoch += 1

            if self.validate:
                sample_f1,_,_,_ = self.eval(model=model, dataloader=val_dataloader, weight=False)
                self.writer.add_scalar('Val_'+section,sample_f1, epoch_i)
                if self.early_stop:
                    if sample_f1 < last_epoch_f1:
                        print("The best epoch is", epoch_i-1)
                        break

                    self.save_checkpoint(model, optimizer, section)
                    last_epoch_f1 = sample_f1
            # test_f1,_,_,_ = self.eval(model=model, dataloader=test_dataloader)
            # self.writer.add_scalar('Test_'+section, test_f1, epoch_i)
            print('--------')
            print('Train eval epoch ', self.global_epoch)
            self.eval(model=model, dataloader=dataloader, weight=False)
            # self.writer.add_scalar('Test_'+section, test_f1, epoch_i)
        
        if self.early_stop == False:
            self.save_checkpoint(model, optimizer, section)

        model, optimizer = self.resume_from_checkpoint(section)

        return model, optimizer
    
    def thresh_pred_prob(self,probabilities):
        # probabilities are only for event probabilities
        
        pred = []
        for i in range(len(probabilities)):
            if probabilities[i] >= self.threshold:
                pred.append(1)
            else: 
                pred.append(0)
        return pred


    def eval(self, model, dataloader, weight):
        # print('Training eval epoch ', self.global_epoch)
        # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        probs = np.zeros((0, self.num_classes))
        preds = []
        labels = []
        file_names = []

        model.eval()
        for i, batch in enumerate(dataloader):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()
            b_file_name = list(batch["file"][0])

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
            predictions = (
                np.argmax(logits.detach().cpu().numpy(),
                          axis=1).flatten().tolist()
            )
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            probs = np.concatenate((probs, probabilities), axis=0)
            preds += predictions
            labels += label_ids
            file_names += b_file_name
        
        # perform thresholding to adjust the preds
        preds = self.thresh_pred_prob(probs[:,1].flatten().tolist())
        model_pred_df = pd.DataFrame({"File Name":file_names, "Labels":labels, "Predicted": preds})
        model_pred_df = model_pred_df.drop_duplicates()
        # model_pred_df.to_csv('model_pred_df.csv')
        if weight:
            sample_f1 = sample_weighted_f1_score(weight_file=self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        else:
            sample_f1 = weight_evaluation.sample_weighted_f1_score(weight_file=self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
            
            
        print("Sample F1 : {0: .6f} ".format(sample_f1))
        accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
        print("Accuracy:", accuracy)
        return sample_f1, probs, labels, file_names

    def ensemble_eval(
        self, view1_model, view2_model, dataloader1, dataloader2, type_df
    ):
        # store the prob, preds and labels
        view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
            view1_model, dataloader1, weight=True)
        view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
            view2_model, dataloader2, weight=True)

        # first check if the view1 and view2 labels are the same:
        if view1_file_names == view2_file_names:
            print("The files names for findings and impressiosn are the same")
        else:
            print("!!! The files names for findings and impressiosn are not the same !!!")

        # combine the probabilities together
        # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
        probabilities = np.add(view1_probs, view2_probs)
        avg_probs = np.divide(probabilities, 2)
        pred_labels = self.thresh_pred_prob(avg_probs[:,1].flatten().tolist())
        model_pred_df = pd.DataFrame({"File Name":view1_file_names, "Labels":view1_labels, "Predicted": pred_labels})
        model_pred_df = model_pred_df.drop_duplicates()
        print('********************')
        print("Results for ensemble")

        # accuracy:
        ensemble_f1 = sample_weighted_f1_score(weight_file=self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print("Sample F1 : {0: .6f} ".format(ensemble_f1))
        ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(view1_labels)
        print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
        self.writer.add_scalar("Accuracy" + "_Ensemble" +
                               "_"+type_df, ensemble_accuracy, self.global_epoch)

        return ensemble_f1, view1_f1, view2_f1
    
    def eval_together(self, model, dataloader, section="combined_sections", model_pred=True):

        print(section)
        
        model_f1, probs, labels, file_names = self.eval(model, dataloader, weight=True)
        print(probs)
        probs = probs[:,1] # event only

        print("Results for", section, "view")

        # take the average of the probabilities
        df = pd.DataFrame({"File Name":file_names, "Labels":labels, "Probabilities":probs})
        print('Shape of df - eval together:', df.shape)
        df.to_csv('combine_test.csv')
        grouped_df = df.groupby(['File Name', "Labels"], as_index = False)['Probabilities'].mean()
        print(grouped_df.head())
        # print(grouped_df.head())
        grouped_df['Predicted'] = 0
        for i in range(grouped_df.shape[0]):
            if grouped_df.loc[i,'Probabilities'] >= self.threshold:
                grouped_df.loc[i,'Predicted'] = 1
            else:
                grouped_df.loc[i,'Predicted'] = 0

        
        # remove duplicates
        model_pred_df = grouped_df.drop_duplicates()    
        # accuracy:
        accuracy = np.sum(np.array(model_pred_df["Predicted"]) == np.array(model_pred_df["Labels"])) / len(model_pred_df["Labels"])
        print("Accuracy : {0: .6f} ".format(accuracy))
        
        ensemble_f1 = sample_weighted_f1_score(weight_file=self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
        print("Sample F1 : {0: .6f} ".format(ensemble_f1))
        
        return ensemble_f1,model_f1,0


    def record_results(self, view1_model, view2_model, dataloader1, dataloader2, type='test', combine=False):
        if combine==False:
            ensemble_acc, view1_acc, view2_acc = self.ensemble_eval(
                view1_model=view1_model,
                view2_model=view2_model,
                dataloader1=dataloader1,
                dataloader2=dataloader2,
                type_df=type,
            )

            # record save the file:
            results = pd.DataFrame({
                "Ensemble accuracy": ensemble_acc,
                "View1 accuracy": view1_acc,
                "View2 accuracy": view2_acc,
                'Val/Test': type,
            }, index=[0])

            self.results_df = pd.concat([self.results_df, results])
            self.results_df.to_csv(os.path.join(self.logdir,  self.target + "_" + 'results.csv'))
        
        else:
            ensemble_acc, view1_acc, view2_acc = self.eval_together(
                model=view1_model,
                dataloader=dataloader1,
            )

            # record save the file:
            results = pd.DataFrame({
                "Ensemble accuracy": ensemble_acc,
                "View1 accuracy": view1_acc,
                "View2 accuracy": view2_acc,
                'Val/Test': type,
            }, index=[0])

            self.results_df = pd.concat([self.results_df, results])
            self.results_df.to_csv(os.path.join(self.logdir,  self.target + "_" + 'results.csv'))



    def ensemble(self):

        print("Finetuning the models")
        print("Shape of train_df:", self.train_df.shape)

        # Finetune each view
        view1_model, _ = self.train(
           model=self.init_model1,
           df=self.train_df,
           val_dataloader=self.view1_val_dataloader,
           test_dataloader=self.view2_test_dataloader,
           section=self.view1,
           optimizer=self.view1_optimizer
        )
        self.global_epoch = 0
        # view2_model, _ = self.train(
        #     model=self.init_model2,
        #     df=self.train_df,
        #     val_dataloader=self.view2_val_dataloader,
        #     test_dataloader=self.view2_test_dataloader,
        #     section=self.view2,
        #     optimizer=self.view2_optimizer
        # )

        if self.validate:
            print("---- record val results ---")
            self.record_results(
                view1_model=view1_model,
                view2_model=view2_model,
                dataloader1=self.view1_val_dataloader,
                dataloader2=self.view2_val_dataloader, type='val')

        #print("---- record test results ---")
        #self.record_results(
        #    view1_model=view1_model,
        #    view2_model=view2_model,
        #    dataloader1=self.view1_test_dataloader,
        #    dataloader2=self.view2_test_dataloader, type='test')
    
    def ensemble_combine(self, section):
        # Finetune each view
        view1_model, _ = self.train(
            model=self.init_model1,
            df=self.train_df,
            val_dataloader=self.view1_val_dataloader,
            test_dataloader=self.view1_test_dataloader,
            section=section,
            optimizer=self.view1_optimizer,
            epochs=self.num_epochs
        )

        if self.validate:
            self.record_results(
                view1_model=view1_model,
                view2_model=None,
                dataloader1=self.view1_val_dataloader,
                dataloader2=None, type='val', combine=True)

        self.record_results(
            view1_model=view1_model,
            view2_model=None,
            dataloader1=self.view1_test_dataloader,
            dataloader2=None, type='test', combine=True)
        



if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--labeled-pickle",
        type=str,
        help="Stores the cleaned labeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
        default='Data/two_train_test_background_combined.pkl'
    )

    argparser.add_argument(
        "--unlabeled-pickle",
        type=str,
        help="Stores the cleaned unlabeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
        default='Data/cotrain_new_samples.pkl'
    )

    argparser.add_argument(
        "--weighted-pickle",
        type=str,
        help="Stores weighted and cleaned labeled data",
        default='Data/weighted_aug.pkl'
    )

    argparser.add_argument(
        "--use-weight",
        action="store_true",
        help="If true, use sample weights for training",
    )

    argparser.add_argument(
        "--combine-sections",
        action="store_true",
        help="If true, combine sections and eval one model",
    )

    argparser.add_argument(
        "--max-length",
        type=int,
        help="The max number of tokens per sequence",
        default=512,
    )
    argparser.add_argument("--random-seed", type=int, default=0)
    argparser.add_argument("--val-rand", type=int, default=0)
    argparser.add_argument("--test-split", type=float, default=0.4)

    # Model Settings:
    # argparser.add_argument("--model-name", type=str,
    #                        default="distilbert-base-cased")
    argparser.add_argument("--model-name", type=str,
                           default="UFNLP/gatortron-base")

    argparser.add_argument("--view1-name", type=str, default="findings")
    argparser.add_argument("--view2-name", type=str, default="impressions")

    # Training
    argparser.add_argument("--batch-size", type=int, default=16)
    argparser.add_argument("--num_classes", type=int, default=2)
    argparser.add_argument("--learning-rate", type=float, default=5e-5)
    argparser.add_argument("--num-epochs", type=int, default=3)
    argparser.add_argument("--use-history", action='store_true')
    argparser.add_argument("--target", type=str, default="mass_label")
    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--validate", action='store_true')
    argparser.add_argument("--weight-file-path", type=str, default='Resources/ir2_weight_by_report.csv')

    # Path to tensorboard and csv of results:
    argparser.add_argument(
        "--logdir",
        type=str,
        default="log/",
        help="Path to save results to",
    )

    argparser.add_argument(
        "--early-stop",
        action="store_true",
        help="If true, early stop using validation",
    )

    args = argparser.parse_args()
    command_str = '> Command:', ' '.join(sys.argv)

    # Set device:
    device = torch.device("cuda")

    # load pd.dataframe
    if args.use_weight:
        labeled_df = pd.read_pickle(args.weighted_pickle)
    else:
        labeled_df = pd.read_pickle(args.labeled_pickle)
        print(args.labeled_pickle)

    

    # test_df = pd.read_pickle('Data/new_test_background_mod.pkl')
    test_df = pd.read_pickle('Data/ir2_background.pkl')

    torch.cuda.empty_cache()

    run = 0

    if args.validate:
        test_df, val_df = train_test_split(test_df, test_size=0.4, random_state=args.val_rand)
    else:
        val_df = test_df
    
    print(test_df.shape)
    # train_df = pd.concat([labeled_df,test_df]) # augment labeled data with test df
    train_df = labeled_df
    print(train_df.shape)

    # Set the seed value all over the place to make this reproducible.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    os.makedirs(args.logdir+"_"+str(run), exist_ok=True)

    view2_name = 'impressions'
    if args.use_history:
        view2_name='impressions_back'

    if args.combine_sections:
        train_df2 = train_df.copy()
        train_df['combined_sections'] = train_df['findings']

        if args.use_history:
            train_df2['combined_sections'] = train_df['impressions']
        else:
            train_df2['combined_sections'] = train_df['impressions_back']

        train_df = pd.concat([train_df, train_df2])
        train_df = train_df.reset_index()

        val_df2 = val_df.copy()
        val_df['combined_sections'] = val_df['findings']
        if args.use_history:
            val_df2['combined_sections'] = val_df['impressions']
        else:
            val_df2['combined_sections'] = val_df['impressions_back']
        val_df = pd.concat([val_df, val_df2])
        val_df = val_df.reset_index()

        test_df2 = test_df.copy()
        test_df['combined_sections'] = test_df['findings']
        if args.use_history:
            test_df2['combined_sections'] = test_df['impressions']
        else:
            test_df2['combined_sections'] = test_df['impressions_back']
        test_df = pd.concat([test_df, test_df2])
        test_df = test_df.reset_index()
    
    
    train_df['aggressive_label'] = train_df['aggressive_label'].replace(
        2, 1)
    # test_df['aggressive_label'] = test_df['aggressive_label'].replace(
    #     2, 1)
    # val_df['aggressive_label'] = val_df['aggressive_label'].replace(
    #     2, 1)
    
    if args.combine_sections:

        # initialize co-training model:
        method = BaseModel(
            model_name=args.model_name,
            view1_name="combined_sections",
            view2_name=view2_name,
            logdir=args.logdir+"_"+str(run),
            train_df=train_df,
            test_df=test_df,
            val_df=val_df,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            target=args.target,
            threshold=args.threshold,
            use_weight=args.use_weight,
            validate=args.validate,
            weight_file_path=args.weight_file_path,
            args=command_str
        )

        method.ensemble_combine(section="combined_sections", early_stop=args.early_stop)

    else:

        method = BaseModel(
            model_name=args.model_name,
            view1_name=args.view1_name,
            view2_name=view2_name,
            logdir=args.logdir+"_"+str(run),
            train_df=train_df,
            test_df=test_df,
            val_df=val_df,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            target=args.target,
            threshold=args.threshold,
            use_weight=args.use_weight,
            validate=args.validate,
            weight_file_path=args.weight_file_path,
            early_stop=args.early_stop,
            args=command_str
        )
        
        method.ensemble()

