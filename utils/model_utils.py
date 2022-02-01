import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch 
import transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
def get_automodel(rate=0.0):
  auto_model = transformers.BertForSequenceClassification.from_pretrained(
          "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
          num_labels = 2, # The number of output labels--2 for binary classification.
                          # You can increase this for multi-class tasks.   
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
  )
  auto_model.dropout = torch.nn.Dropout(p=rate, inplace=False)
  auto_model.to(device)
  return auto_model

# model definition
class BinaryClassificationBERT(torch.nn.Module):
    # define model elements
    def __init__(self, dropout_rate=0.1, num_labels=2):
        super(BinaryClassificationBERT, self).__init__()
        self.base = self._get_base_model()
        self.dropout = torch.nn.Dropout(p=dropout_rate, inplace=False)
        self.cfs = torch.nn.Linear(in_features=768, out_features=num_labels, bias=True)

    def _get_base_model(self):
        from transformers import BertModel, BertConfig
        configuration = BertConfig()
        model = BertModel(configuration)
        return model

    # forward propagate input
    def forward(self, X, token_type_ids, attention_mask):
        X = self.base(X, token_type_ids=token_type_ids, 
                         attention_mask=attention_mask)
        #print(X.last_hidden_state.shape, X.pooler_output.shape)
        X = X.pooler_output
        #X = self.dropout(X.pooler_output)
        X = self.cfs(X)
        #print(X.shape)
        return X
