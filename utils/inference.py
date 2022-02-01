import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch 
import transformers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_inference(model, prediction_dataloader, filename='submission.csv'):
  """
  Run inference and write the result to filename
  """
  model.eval()    
  f = open("submission.csv", "w")
  f.write("id,target")
  for batch in prediction_dataloader:
      
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      id, b_input_ids, b_input_mask = batch
      
      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():        

          # Forward pass, calculate logit predictions.
          # This will return the logits rather than the loss because we have
          # not provided labels.
          # token_type_ids is the same as the "segment ids", which 
          # differentiates sentence 1 and 2 in 2-sentence tasks.
          # The documentation for this `model` function is here: 
          # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
          outputs = model(b_input_ids, 
                          token_type_ids=None, 
                          attention_mask=b_input_mask)
      
      # Get the "logits" output by the model. The "logits" are the output
      # values prior to applying an activation function like the softmax.
      logits = outputs[0]

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      logits = np.argmax(logits, axis=1).flatten()

      f.write("\n"+str(id.cpu().numpy()[0]) + "," + str(logits.item()))

