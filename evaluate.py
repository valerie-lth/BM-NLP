# import os
# import pandas as pd
# import numpy as np
# import transformers
# import torch
# import torch.nn as nn
# # from torch.optim import AdamW
# from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from torchsummary import summary
# from tqdm import tqdm
# import argparse
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold


# # from transformers import AdamW
# import time
# import datetime
# import random
# from sklearn.metrics import (
#     confusion_matrix,
#     auc,
#     roc_auc_score,
#     roc_curve,
#     precision_recall_curve,
#     f1_score,
# )
# import math
# import sys
# from dateutil import tz

# from data_processing import organize_labeled_data, organize_unlabeled_data
# from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset
# from weight_evaluation_weighted import estimate_weighted_f1_score, sample_weighted_f1_score, sample_weighted_precision, sample_weighted_recall, estimate_threshold
# import weight_evaluation
# from torch.utils.tensorboard import SummaryWriter


# class Logger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#         self.encoding = "UTF-8"

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#         self.log.flush()

#     def flush(self):
#         self.terminal.flush()
#         self.log.flush



# class Evaluate:
#     def __init__(
#         self,
#         model_name: str,
#         logdir: str,
#         test_df,
#         val_df,
#         view1_name: str,
#         view2_name: str,
#         max_length: int,
#         batch_size: int,
#         num_classes: int,
#         threshold: float,
#         target: str,
#         model1: str,
#         model2: str,
#         thresh:float,
#     ):
#         tzone = tz.gettz("America/Edmonton")
#         self.timestamp = (
#             datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")
#         )

#         self.model_name = model_name
#         self.view1 = view1_name
#         self.view2 = view2_name

#         # writer to log information:
#         self.logdir = logdir
#         self.logger = Logger(os.path.join(
#             self.logdir, self.timestamp + ".log"))
#         sys.stdout = self.logger
#         sys.stderr = self.logger
        
#         # assign all of the various data sets that we need
#         self.test_df = test_df
#         self.val_df = val_df
#         # self.weight_file_path = 'Resources/ir2_weight_by_report.csv'
#         # self.weight_file_path = 'Resources/ir2_weight_by_patient.csv'
#         self.weight_file_path = '/apps/data/AHS_project/ir2_nonmal_weight.csv'

#         # Set training parameters of each separate model:
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#         self.target = target
#         self.threshold = threshold

#         # threshold for recall level
#         self.thresh = thresh

#         # Load datasets
#         self.view1_test_dataloader = self.load_dataset(
#             section=self.view1,
#             df=self.test_df.reset_index(drop=True),
#             labeled=True,
#             shuffle=False,
#         )

#         self.view2_test_dataloader = self.load_dataset(
#             section=self.view2,
#             df=self.test_df.reset_index(drop=True),
#             labeled=True,
#             shuffle=False,
#         )

#         self.view1_val_dataloader = self.load_dataset(
#             section=self.view1,
#             df=self.val_df.reset_index(drop=True),
#             labeled=True,
#             shuffle=False,
#         )

#         self.view2_val_dataloader = self.load_dataset(
#             section=self.view2,
#             df=self.val_df.reset_index(drop=True),
#             labeled=True,
#             shuffle=False,
#         )




#         # Both views are initialized from the same model
#         self.model1 = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name,
#             num_labels=self.num_classes,  # The number of output labels
#             # Whether the model returns attentions weights.
#             output_attentions=False,
#             # Whether the model returns all hidden-states.
#             output_hidden_states=False,
#         ).cuda()


#         self.model2 = AutoModelForSequenceClassification.from_pretrained(
#             self.model_name,
#             num_labels=self.num_classes,  # The number of output labels
#             # Whether the model returns attentions weights.
#             output_attentions=False,
#             # Whether the model returns all hidden-states.
#             output_hidden_states=False,
#         ).cuda()

#         checkpoint = torch.load(model1)
#         self.model1.load_state_dict(checkpoint['model_state_dict'], strict=False)
#         self.model1.cuda()

#         checkpoint = torch.load(model2)
#         self.model2.load_state_dict(checkpoint['model_state_dict'], strict=False)
#         self.model2.cuda()

#     def load_dataset(self, section, df, labeled=True, other_section="", shuffle=True, use_weight=False):
#         # load data into dataloader + tokenize
#         tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name)
#         if labeled:
#             dataset = RadiologyLabeledDataset(
#                 tokenizer,
#                 max_length=self.max_length,
#                 df=df,
#                 target=self.target,
#                 view_name=section,
#                 use_weight=use_weight
#             )
#             dataloader = DataLoader(
#                 dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
#             )
#         return dataloader
    
#     def thresh_pred_prob(self,probabilities):
#         # probabilities are only for event probabilities
        
#         pred = []
#         for i in range(len(probabilities)):
#             if probabilities[i] >= self.threshold:
#                 pred.append(1)
#             else: 
#                 pred.append(0)
#         return pred


#     def eval(self, model, dataloader, save=True):

#         # softmax function that we need for metric calculations:
#         softmax = nn.Softmax(dim=-1)

#         # store the prob, preds and labels
#         probs = np.zeros((0, self.num_classes))
#         preds = []
#         labels = []
#         file_names = []

#         model.eval()
#         for i, batch in enumerate(dataloader):
#             b_input_ids = batch["ids"].cuda()
#             b_input_mask = batch["mask"].cuda()
#             b_labels = batch["target"].cuda()
#             b_file_name = list(batch["file"][0])

#             with torch.no_grad():
#                 # Forward pass, calculate logit predictions.
#                 result = model(
#                     b_input_ids,
#                     attention_mask=b_input_mask,
#                     labels=b_labels,
#                     return_dict=True,
#                 )

#             logits = result.logits

#             # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
#             probabilities = (softmax(logits).detach().cpu().numpy())
#             predictions = (
#                 np.argmax(logits.detach().cpu().numpy(),
#                           axis=1).flatten().tolist()
#             )
#             label_ids = b_labels.cpu().numpy().flatten().tolist()

#             probs = np.concatenate((probs, probabilities), axis=0)
#             preds += predictions
#             labels += label_ids
#             file_names += b_file_name
        
#         # perform thresholding to adjust the preds
#         preds = self.thresh_pred_prob(probs[:,1].flatten().tolist())
#         model_pred_df = pd.DataFrame({"File Name":file_names, "Labels":labels, "Predicted": preds})
#         print(model_pred_df.shape)
#         # model_pred_df = model_pred_df.drop_duplicates()
#         print(model_pred_df.shape)
#         model_pred_df.to_csv('model_pred_df.csv')
#         sample_f1 =  weight_evaluation.sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample F1 : {0: .6f} ".format(sample_f1))
#         accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
#         print("Accuracy:", accuracy)
#         sample_recall =  weight_evaluation.sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print('Recall:', sample_recall)

#         weight_evaluation.estimate_threshold(self.weight_file_path,model_pred_df=model_pred_df, model_event_prob=probs[:,1].flatten().tolist(), target=self.target, thresh=self.thresh)
#         return sample_f1, probs[:, 1], labels, file_names

#     def ensemble_max(self):
#         '''If one of the model predicts as yes, use that prediction'''
#         # store the prob, preds and labels
#         view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
#             model=self.model1, dataloader=self.view1_test_dataloader)
#         view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
#             model=self.model2, dataloader=self.view2_test_dataloader)

#         # first check if the view1 and view2 labels are the same:
#         if view1_file_names == view2_file_names:
#             print("The files names are the same")
#         else:
#             print("Check code")

#         # create masks for probability greater than threshold
#         # mask = (view1_probs >= self.threshold) | (view2_probs >= self.threshold)
#         # final_probs = np.where(mask, np.maximum(view1_probs, view2_probs), np.maximum(view1_probs, view2_probs))

#         # pred_labels = self.thresh_pred_prob(final_probs[:,1].flatten().tolist())
#         # model_pred_df = pd.DataFrame({"File Name":view1_file_names, 
#         #                               "Labels":view1_labels, 
#         #                               "Predicted": pred_labels, 
#         #                               "Probabilities":final_probs[:,1].flatten().tolist(),
#         #                               "Findings_prob": view1_probs[:,1].flatten().tolist(),
#         #                               "Impression_prob":view2_probs[:,1].flatten().tolist(),
#         #                               })
    
#         final_probs = np.maximum(view1_probs, view2_probs)

#         pred_labels = self.thresh_pred_prob(final_probs)
#         model_pred_df = pd.DataFrame({"File Name":view1_file_names, 
#                                       "Labels":view1_labels, 
#                                       "Predicted": pred_labels, 
#                                       "Probabilities":final_probs,
#                                       "Findings_prob": view1_probs,
#                                       "Impression_prob":view2_probs,
#                                       })
#         model_pred_df.to_csv(self.target+'_ir2_predictions_ensmeble_max_05.csv')
#         print("Results for ensemble")
#         # accuracy:
#         ensemble_f1 = sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample F1 : {0: .6f} ".format(ensemble_f1))
#         ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
#             view1_labels
#         )
#         print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
#         ensemble_recall =  sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample Recall : {0: .6f} ".format(ensemble_recall))
#         ensemble_precision = (ensemble_f1 * ensemble_recall) / (2 * ensemble_recall - ensemble_f1)
#         print('Precision: ', ensemble_precision)

#         # estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=final_probs[:,1].flatten().tolist(), target=self.target, thresh=self.thresh, visualize=False)
#         estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=final_probs, target=self.target, thresh=self.thresh, visualize=False)


#         # # accuracy:
#         # ensemble_f1 = weight_evaluation.sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         # print("Sample F1 : {0: .6f} ".format(ensemble_f1))
#         # ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
#         #     view1_labels
#         # )
#         # print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
#         # ensemble_recall =  weight_evaluation.sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         # print("Sample Recall : {0: .6f} ".format(ensemble_recall))

#         # weight_evaluation.estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=final_probs[:,1].flatten().tolist(), target=self.target, thresh=self.thresh, visualize=False)

#         return ensemble_f1, view1_f1, view2_f1


#     def ensemble_eval(self):
#         # store the prob, preds and labels
#         view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
#             model=self.model1, dataloader=self.view1_test_dataloader)
#         view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
#             model=self.model2, dataloader=self.view2_test_dataloader)

#         # first check if the view1 and view2 labels are the same:
#         if view1_file_names == view2_file_names:
#             print("The files names are the same")
#         else:
#             print("Check code")

#         # combine the probabilities together
#         # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
#         probabilities = np.add(view1_probs, view2_probs)
#         avg_probs = np.divide(probabilities, 2)
#         pred_labels = self.thresh_pred_prob(avg_probs)
#         model_pred_df = pd.DataFrame({"File Name":view1_file_names, 
#                                       "Labels":view1_labels, 
#                                       "Predicted": pred_labels, 
#                                       "Probabilities":avg_probs,
#                                       "Findings_prob": view1_probs,
#                                       "Impression_prob":view2_probs,
#                                       })
#         model_pred_df = model_pred_df.drop_duplicates()
#         # section_text = self.test_df[['File Name', "findings", "impressions", "sections"]]
#         # analysis_df = pd.merge(model_pred_df,section_text, on=['File Name'],how="inner")
#         # annotations = pd.read_csv('Data/reports_by_patients_0724_2024.csv')
#         # analysis_df = analysis_df.merge(annotations, on=['File Name'])
        
#         # model_pred_df.to_csv(self.target+'_predictions_ensmeble_avg_05.csv')

#         print("Results for ensemble")
#         # accuracy:
#         ensemble_f1 = sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample F1 : {0: .6f} ".format(ensemble_f1))
#         ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
#             view1_labels
#         )
#         print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
#         ensemble_recall =  sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample Recall : {0: .6f} ".format(ensemble_recall))

#         estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=avg_probs, target=self.target, thresh=self.thresh, visualize=False)

#         # # accuracy:
#         # ensemble_f1 = weight_evaluation.sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         # print("Sample F1 : {0: .6f} ".format(ensemble_f1))
#         # ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
#         #     view1_labels
#         # )
#         # print("Accuracy : {0: .6f} ".format(ensemble_accuracy))
#         # ensemble_recall =  weight_evaluation.sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         # print("Sample Recall : {0: .6f} ".format(ensemble_recall))

#         # weight_evaluation.estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=avg_probs[:,1].flatten().tolist(), target=self.target, thresh=self.thresh, visualize=False)

#         return ensemble_f1, view1_f1, view2_f1
    
#     def eval_together(self):

#         # softmax function that we need for metric calculations:
#         softmax = nn.Softmax(dim=-1)
        
#         model_f1, probs, labels, file_names = self.eval(self.model1, self.view1_test_dataloader)
#         print(probs)
#         probs = probs[:,1] # event only

#         # take the average of the probabilities
#         df = pd.DataFrame({"File Name":file_names, "Labels":labels, "Probabilities":probs})
#         print('Shape of df - eval together:', df.shape)
#         df.to_csv('combine_test.csv')
#         grouped_df = df.groupby(['File Name', "Labels"], as_index = False)['Probabilities'].mean()
#         print(grouped_df.head())
#         # print(grouped_df.head())
#         grouped_df['Predicted'] = 0
#         for i in range(grouped_df.shape[0]):
#             if grouped_df.loc[i,'Probabilities'] >= self.threshold:
#                 grouped_df.loc[i,'Predicted'] = 1
#             else:
#                 grouped_df.loc[i,'Predicted'] = 0

        
#         # remove duplicates
#         model_pred_df = grouped_df.drop_duplicates()
#         # model_pred_df.to_csv(self.target+'_predictions.csv')    
#         # accuracy:
#         accuracy = np.sum(np.array(model_pred_df["Predicted"]) == np.array(model_pred_df["Labels"])) / len(model_pred_df["Labels"])
#         print("Accuracy : {0: .6f} ".format(accuracy))
        
#         ensemble_f1 = sample_weighted_f1_score(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample F1 : {0: .6f} ".format(ensemble_f1))

#         ensemble_recall = sample_weighted_recall(self.weight_file_path, model_pred_df=model_pred_df, target=self.target)
#         print("Sample Recall: {0: .6f} ".format(ensemble_recall))

#         estimate_threshold(self.weight_file_path, model_pred_df=model_pred_df, model_event_prob=probs.flatten().tolist(), target=self.target, thresh=self.thresh)

        
        
#         return ensemble_f1,model_f1,0


# if __name__ == "__main__":
#     # parse some arguments that are needed
#     argparser = argparse.ArgumentParser()

#     argparser.add_argument(
#         "--combine-sections",
#         action="store_true",
#         help="If true, combine sections and eval one model",
#     )

#     argparser.add_argument(
#         "--max-length",
#         type=int,
#         help="The max number of tokens per sequence",
#         default=512,
#     )
#     argparser.add_argument("--random-seed", type=int, default=0)
#     argparser.add_argument("--val-rand", type=int, default=0)
#     argparser.add_argument("--test-split", type=float, default=0.4)

#     # Model Settings:
#     argparser.add_argument("--model-name", type=str,
#                            default="distilbert-base-cased")
#     argparser.add_argument("--model1", type=str, help='The path to view1 checkpoint')
#     argparser.add_argument("--model2", type=str, help='The path to view2 checkpoint')
#     argparser.add_argument("--view1-name", type=str, default="findings")
#     argparser.add_argument("--view2-name", type=str, default="impressions")

#     # Training
#     argparser.add_argument("--batch-size", type=int, default=16)
#     argparser.add_argument("--num_classes", type=int, default=2)
#     argparser.add_argument("--target", type=str, default="mass_label")
#     argparser.add_argument("--threshold", type=float, default=0.5)
#     argparser.add_argument("--recall-thresh", type=float, default=0.9)

#     # Path to tensorboard and csv of results:
#     argparser.add_argument(
#         "--logdir",
#         type=str,
#         default="logs/",
#         help="Path to save results to",
#     )
#     argparser.add_argument("--test-data-pkl", type=str, default='Data/post_diag_reports_processed.pkl', help='Data pkl path for inference/evaluation')


#     args = argparser.parse_args()

#     # Set device:
#     device = torch.device("cuda")

#     torch.cuda.empty_cache()

#     # for new data:
#     # test_df = pd.read_pickle('Data/ir2_background_not_none.pkl')
#     test_df = pd.read_pickle(args.test_data_pkl)
#     # test_df = pd.read_pickle('Data/ir_background_pop10_v2.pkl')
#     # test_df = pd.read_pickle('../Cotraining/cleaned.pkl')
#     # test_df = pd.read_pickle('Data/all_ir.pkl')
#     # test_df = pd.read_pickle('Data/ir2_with_1section.pkl')
#     print(test_df.shape)

#     # test_df, val_df = train_test_split(test_df, test_size=0.4, random_state=0)
#     # print(test_df.shape)

#     # Set the seed value all over the place to make this reproducible.
#     random.seed(args.random_seed)
#     np.random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
#     torch.cuda.manual_seed_all(args.random_seed)

#     os.makedirs(args.logdir, exist_ok=True)

#     # if args.combine_sections:

#     #     val_df2 = val_df.copy()
#     #     val_df['combined_sections'] = val_df['findings']
#     #     val_df2['combined_sections'] = val_df['impressions']
#     #     val_df = pd.concat([val_df, val_df2])
#     #     val_df = val_df.reset_index()

#     #     test_df2 = test_df.copy()
#     #     test_df['combined_sections'] = test_df['findings']
#     #     test_df2['combined_sections'] = test_df['impressions']
#     #     test_df = pd.concat([test_df, test_df2])
#     #     test_df = test_df.reset_index()
    
#     # if args.target == 'aggressive_label':
#     #     test_df['aggressive_label'] = test_df['aggressive_label'].replace(
#     #         2, 1)
#     #     val_df['aggressive_label'] = val_df['aggressive_label'].replace(
#     #         2, 1)
    
    
#     if args.combine_sections:

#         # initialize co-training model:
#         method = Evaluate(
#             model_name=args.model_name,
#             view1_name="combined_sections",
#             view2_name=args.view2_name,
#             logdir=args.logdir,
#             test_df=test_df,
#             # val_df=val_df,
#             val_df=test_df,
#             max_length=args.max_length,
#             batch_size=args.batch_size,
#             num_classes=args.num_classes,
#             target=args.target,
#             threshold=args.threshold,
#             model1=args.model1,
#             model2=args.model2,
#             thresh=args.recall_thresh,
#         )

#         method.eval_together()

#     else:

#         method = Evaluate(
#             model_name=args.model_name,
#             view1_name=args.view1_name,
#             view2_name=args.view2_name,
#             logdir=args.logdir,
#             test_df=test_df, # the only difference from evaluate_recall_thresh is to run on test data
#             val_df=test_df, # not needed actually
#             max_length=args.max_length,
#             batch_size=args.batch_size,
#             num_classes=args.num_classes,
#             target=args.target,
#             threshold=args.threshold,
#             model1=args.model1,
#             model2=args.model2,
#             thresh=args.recall_thresh,
#         )
        
#         method.ensemble_eval()
#         # method.ensemble_max()


import os
import sys
import time
import math
import random
import datetime
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dateutil import tz
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset
from weight_evaluation import (
    estimate_weighted_f1_score,
    sample_weighted_f1_score,
    sample_weighted_precision,
    sample_weighted_recall,
    estimate_threshold,
)


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
        self.log.flush()


def normalize_text(x):
    if pd.isna(x) or x is None:
        return ""
    if isinstance(x, str):
        if x.strip().upper() == "N/A":
            return ""
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(str(v) for v in x if pd.notna(v) and v is not None)
    return str(x)


def load_input_dataframe(path):
    if path.endswith(".pkl") or path.endswith(".pickle"):
        df = pd.read_pickle(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file format: {path}")
    return df


class Evaluate:
    def __init__(
        self,
        model_name: str,
        logdir: str,
        test_df,
        val_df,
        view1_name: str,
        view2_name: str,
        max_length: int,
        batch_size: int,
        num_classes: int,
        threshold: float,
        target: str,
        model1: str,
        model2: str,
        thresh: float,
    ):
        tzone = tz.gettz("America/Edmonton")
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")

        self.model_name = model_name
        self.view1 = view1_name
        self.view2 = view2_name
        self.logdir = logdir
        self.target = target
        self.threshold = threshold
        self.thresh = thresh
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = num_classes

        os.makedirs(self.logdir, exist_ok=True)
        self.logger = Logger(os.path.join(self.logdir, self.timestamp + ".log"))
        sys.stdout = self.logger
        sys.stderr = self.logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.test_df = test_df.reset_index(drop=True).copy()
        self.val_df = val_df.reset_index(drop=True).copy()

        self.weight_file_path = "/apps/data/AHS_project/ir2_nonmal_weight.csv"

        # Sanity checks and text cleanup
        for df_name, df in [("test_df", self.test_df), ("val_df", self.val_df)]:
            if "File Name" not in df.columns:
                raise KeyError(f"'File Name' column missing in {df_name}")
            if self.target not in df.columns:
                raise KeyError(f"Target column '{self.target}' missing in {df_name}")

            df[self.target] = pd.to_numeric(df[self.target], errors="raise").astype(int)

            for col in [self.view1, self.view2]:
                if col in df.columns:
                    bad_mask = df[col].isna() | ~df[col].map(lambda x: isinstance(x, str))
                    print(f"{df_name} - {col}: {bad_mask.sum()} non-string / missing values before cleanup")
                    df[col] = df[col].apply(normalize_text)
                else:
                    raise KeyError(f"Column '{col}' missing in {df_name}")

        self.view1_test_dataloader = self.load_dataset(
            section=self.view1,
            df=self.test_df,
            labeled=True,
            shuffle=False,
        )

        self.view2_test_dataloader = self.load_dataset(
            section=self.view2,
            df=self.test_df,
            labeled=True,
            shuffle=False,
        )

        self.view1_val_dataloader = self.load_dataset(
            section=self.view1,
            df=self.val_df,
            labeled=True,
            shuffle=False,
        )

        self.view2_val_dataloader = self.load_dataset(
            section=self.view2,
            df=self.val_df,
            labeled=True,
            shuffle=False,
        )

        self.model1 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        ).to(self.device)

        self.model2 = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        ).to(self.device)

        checkpoint1 = torch.load(model1, map_location="cpu")
        self.model1.load_state_dict(checkpoint1["model_state_dict"], strict=False)
        self.model1.to(self.device)

        checkpoint2 = torch.load(model2, map_location="cpu")
        self.model2.load_state_dict(checkpoint2["model_state_dict"], strict=False)
        self.model2.to(self.device)

    def load_dataset(self, section, df, labeled=True, other_section="", shuffle=True, use_weight=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if labeled:
            dataset = RadiologyLabeledDataset(
                tokenizer=tokenizer,
                max_length=self.max_length,
                df=df,
                target=self.target,
                view_name=section,
                use_weight=use_weight,
            )
        else:
            dataset = RadiologyUnlabeledDataset(
                tokenizer=tokenizer,
                max_length=self.max_length,
                df=df,
                view_name=section,
            )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
        return dataloader

    def thresh_pred_prob(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        return (probabilities >= self.threshold).astype(int).tolist()

    def eval(self, model, dataloader, save=True):
        softmax = nn.Softmax(dim=-1)

        probs = np.zeros((0, self.num_classes))
        preds = []
        labels = []
        file_names = []

        model.eval()
        for _, batch in enumerate(dataloader):
            b_input_ids = batch["ids"].to(self.device)
            b_input_mask = batch["mask"].to(self.device)
            b_labels = batch["target"].to(self.device)
            b_file_names = [str(x) for x in batch["file"]]

            with torch.no_grad():
                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits
            probabilities = softmax(logits).detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            probs = np.concatenate((probs, probabilities), axis=0)
            labels.extend(label_ids)
            file_names.extend(b_file_names)

        event_probs = probs[:, 1]
        preds = self.thresh_pred_prob(event_probs)

        model_pred_df = pd.DataFrame({
            "File Name": file_names,
            "Labels": labels,
            "Predicted": preds,
            "Probabilities": event_probs,
        })

        print("Prediction dataframe shape:", model_pred_df.shape)

        sample_f1 = sample_weighted_f1_score(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample F1 : {0:.6f}".format(sample_f1))

        accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
        print("Accuracy:", accuracy)

        sample_recall = sample_weighted_recall(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Recall:", sample_recall)

        estimate_threshold(
            self.weight_file_path,
            model_pred_df=model_pred_df,
            model_event_prob=event_probs,
            target=self.target,
            thresh=self.thresh,
            visualize=True,
        )

        return sample_f1, event_probs, labels, file_names

    def ensemble_max(self):
        view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
            model=self.model1, dataloader=self.view1_test_dataloader
        )
        view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
            model=self.model2, dataloader=self.view2_test_dataloader
        )

        if view1_file_names == view2_file_names:
            print("The file names are the same")
        else:
            raise ValueError("File name order differs between findings and impressions dataloaders.")

        final_probs = np.maximum(view1_probs, view2_probs)
        pred_labels = self.thresh_pred_prob(final_probs)

        model_pred_df = pd.DataFrame({
            "File Name": view1_file_names,
            "Labels": view1_labels,
            "Predicted": pred_labels,
            "Probabilities": final_probs,
            "Findings_prob": view1_probs,
            "Impression_prob": view2_probs,
        })

        out_csv = os.path.join(self.logdir, f"{self.target}_predictions_ensemble_max.csv")
        model_pred_df.to_csv(out_csv, index=False)

        print("Results for ensemble max")
        ensemble_f1 = sample_weighted_f1_score(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample F1 : {0:.6f}".format(ensemble_f1))

        ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(view1_labels)
        print("Accuracy : {0:.6f}".format(ensemble_accuracy))

        ensemble_recall = sample_weighted_recall(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample Recall : {0:.6f}".format(ensemble_recall))

        ensemble_precision = sample_weighted_precision(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Precision : {0:.6f}".format(ensemble_precision))

        estimate_threshold(
            self.weight_file_path,
            model_pred_df=model_pred_df,
            model_event_prob=final_probs,
            target=self.target,
            thresh=self.thresh,
            visualize=True,
        )

        return ensemble_f1, view1_f1, view2_f1

    def ensemble_eval(self):
        view1_f1, view1_probs, view1_labels, view1_file_names = self.eval(
            model=self.model1, dataloader=self.view1_test_dataloader
        )
        view2_f1, view2_probs, view2_labels, view2_file_names = self.eval(
            model=self.model2, dataloader=self.view2_test_dataloader
        )

        if view1_file_names == view2_file_names:
            print("The file names are the same")
        else:
            raise ValueError("File name order differs between findings and impressions dataloaders.")

        avg_probs = (view1_probs + view2_probs) / 2.0
        pred_labels = self.thresh_pred_prob(avg_probs)

        model_pred_df = pd.DataFrame({
            "File Name": view1_file_names,
            "Labels": view1_labels,
            "Predicted": pred_labels,
            "Probabilities": avg_probs,
            "Findings_prob": view1_probs,
            "Impression_prob": view2_probs,
        })

        out_csv = os.path.join(self.logdir, f"{self.target}_predictions_ensemble_avg.csv")
        model_pred_df.to_csv(out_csv, index=False)

        print("Results for ensemble average")
        ensemble_f1 = sample_weighted_f1_score(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample F1 : {0:.6f}".format(ensemble_f1))

        ensemble_accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(view1_labels)
        print("Accuracy : {0:.6f}".format(ensemble_accuracy))

        ensemble_recall = sample_weighted_recall(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample Recall : {0:.6f}".format(ensemble_recall))

        ensemble_precision = sample_weighted_precision(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Precision : {0:.6f}".format(ensemble_precision))

        estimate_threshold(
            self.weight_file_path,
            model_pred_df=model_pred_df,
            model_event_prob=avg_probs,
            target=self.target,
            thresh=self.thresh,
            visualize=True,
        )

        return ensemble_f1, view1_f1, view2_f1

    def eval_together(self):
        model_f1, probs, labels, file_names = self.eval(self.model1, self.view1_test_dataloader)

        df = pd.DataFrame({
            "File Name": file_names,
            "Labels": labels,
            "Probabilities": probs,
        })

        print("Shape of df - eval together:", df.shape)

        grouped_df = df.groupby(["File Name", "Labels"], as_index=False)["Probabilities"].mean()
        grouped_df["Predicted"] = (grouped_df["Probabilities"] >= self.threshold).astype(int)

        model_pred_df = grouped_df.drop_duplicates()

        accuracy = np.sum(
            np.array(model_pred_df["Predicted"]) == np.array(model_pred_df["Labels"])
        ) / len(model_pred_df["Labels"])
        print("Accuracy : {0:.6f}".format(accuracy))

        ensemble_f1 = sample_weighted_f1_score(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample F1 : {0:.6f}".format(ensemble_f1))

        ensemble_recall = sample_weighted_recall(
            self.weight_file_path, model_pred_df=model_pred_df, target=self.target
        )
        print("Sample Recall: {0:.6f}".format(ensemble_recall))

        estimate_threshold(
            self.weight_file_path,
            model_pred_df=model_pred_df,
            model_event_prob=model_pred_df["Probabilities"].to_numpy(),
            target=self.target,
            thresh=self.thresh,
            visualize=True,
        )

        return ensemble_f1, model_f1, 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--combine-sections",
        action="store_true",
        help="If true, combine sections and eval one model",
    )
    argparser.add_argument("--max-length", type=int, default=512)
    argparser.add_argument("--random-seed", type=int, default=0)
    argparser.add_argument("--val-rand", type=int, default=0)
    argparser.add_argument("--test-split", type=float, default=0.4)

    argparser.add_argument("--model-name", type=str, default="distilbert-base-cased")
    argparser.add_argument("--model1", type=str, required=True, help="Path to view1 checkpoint")
    argparser.add_argument("--model2", type=str, required=True, help="Path to view2 checkpoint")
    argparser.add_argument("--view1-name", type=str, default="findings")
    argparser.add_argument("--view2-name", type=str, default="impressions")

    argparser.add_argument("--batch-size", type=int, default=16)
    argparser.add_argument("--num_classes", type=int, default=2)
    argparser.add_argument("--target", type=str, default="mass_label")
    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--recall-thresh", type=float, default=0.9)

    argparser.add_argument("--logdir", type=str, default="logs/", help="Path to save results to")
    argparser.add_argument(
        "--test-data-pkl",
        type=str,
        default="Data/post_diag_reports_processed.pkl",
        help="Input dataframe path (.pkl or .csv)",
    )

    args = argparser.parse_args()

    torch.cuda.empty_cache()

    test_df = load_input_dataframe(args.test_data_pkl)
    print(test_df.shape)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    os.makedirs(args.logdir, exist_ok=True)

    if args.combine_sections:
        if "combined_sections" not in test_df.columns:
            raise KeyError(
                "combine_sections=True but column 'combined_sections' does not exist in input dataframe."
            )

        method = Evaluate(
            model_name=args.model_name,
            view1_name="combined_sections",
            view2_name=args.view2_name,
            logdir=args.logdir,
            test_df=test_df,
            val_df=test_df,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            target=args.target,
            threshold=args.threshold,
            model1=args.model1,
            model2=args.model2,
            thresh=args.recall_thresh,
        )

        method.eval_together()

    else:
        method = Evaluate(
            model_name=args.model_name,
            view1_name=args.view1_name,
            view2_name=args.view2_name,
            logdir=args.logdir,
            test_df=test_df,
            val_df=test_df,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            target=args.target,
            threshold=args.threshold,
            model1=args.model1,
            model2=args.model2,
            thresh=args.recall_thresh,
        )

        method.ensemble_eval()
        # method.ensemble_max()
