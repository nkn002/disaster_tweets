import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch 
import transformers
import time
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
  
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def run_automodel(auto_model, optimizer, criterion, epochs, train_dataloader, validation_dataloader, scheduler=None, verbose=False):
  loss_values = []
  val_loss_values = []
  for epoch_i in range(0, epochs):
      #Training
      if verbose:
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
      t0 = time.time()
      total_loss = 0
      auto_model.train()

      for step, batch in enumerate(train_dataloader):
          # if step % 1000 == 0 and not step == 0:
          #     elapsed = format_time(time.time() - t0)
              
          #     # Report progress.
          #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

          # Unpack this training batch from our dataloader. 
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
          #Clear previous grad value
          auto_model.zero_grad()        
          # Perform a forward pass
          outputs = auto_model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask, 
                      labels=b_labels)
          loss = outputs[0] # The call to `auto_model` returns a tuple
          total_loss += loss.item()
          # Perform a backward pass to calculate the gradients.
          loss.backward()
          #torch.nn.utils.clip_grad_norm_(auto_model.parameters(), 1.0)

          # Update parameters and take a step using the computed gradient.
          optimizer.step()

          if scheduler != None:
            scheduler.step()

      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)            
      # Store the loss value for plotting the learning curve.
      loss_values.append(avg_train_loss)

      if verbose:
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
          
      #Validation
        print("")
        print("Running Validation...")
      val_total_loss = 0
      t0 = time.time()
      auto_model.eval()

      # Tracking variables 
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      for batch in validation_dataloader:
          # Add batch to GPU
          batch = tuple(t.to(device) for t in batch)
          # Unpack the inputs from our dataloader
          b_input_ids, b_input_mask, b_labels = batch

          with torch.no_grad():        
              outputs = auto_model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask)
          # "logits": output values prior to applying an activation function like the softmax.
          logits = outputs[0]
          val_loss = criterion(logits, b_labels)
          val_total_loss += val_loss.item()

          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          tmp_eval_accuracy = flat_accuracy(logits, label_ids)
          # Accumulate the total accuracy.
          eval_accuracy += tmp_eval_accuracy

          # Track the number of batches
          nb_eval_steps += 1

      avg_val_loss = val_total_loss / len(validation_dataloader)            
      val_loss_values.append(avg_val_loss)

      # Report the final accuracy for this validation run.
      if verbose:
        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
  return loss_values, val_loss_values, eval_accuracy/nb_eval_steps

def run_repmodel(model, optimizer, criterion, epochs, train_dataloader, validation_dataloader):
  loss_values = []
  val_loss_values = []
  for epoch_i in range(0, epochs):
      #Training
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      t0 = time.time()
      total_loss = 0

      model.train()
      for step, batch in enumerate(train_dataloader):
          if step % 1000 == 0 and not step == 0:
              elapsed = format_time(time.time() - t0)
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

          b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

          model.zero_grad()        
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          
          loss = criterion(outputs, b_labels)
          total_loss += loss.item()

          # Perform a backward pass to calculate the gradients.
          loss.backward()
          #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()

      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)            
      
      # Store the loss value for plotting the learning curve.
      loss_values.append(avg_train_loss)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
          
      # Validation

      print("")
      print("Running Validation...")

      val_total_loss = 0
      t0 = time.time()
      model.eval()

      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      for batch in validation_dataloader:
          
          # Add batch to GPU
          batch = tuple(t.to(device) for t in batch)
          
          # Unpack the inputs from our dataloader
          b_input_ids, b_input_mask, b_labels = batch
          
          # Telling the model not to compute or store gradients, saving memory and
          # speeding up validation
          with torch.no_grad():        
              outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask)
          
          logits = outputs          
          val_loss = criterion(logits, b_labels)
          val_total_loss += val_loss.item()
          
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()

          tmp_eval_accuracy = flat_accuracy(logits, label_ids)
          # Accumulate the total accuracy.
          eval_accuracy += tmp_eval_accuracy

          # Track the number of batches
          nb_eval_steps += 1

      avg_val_loss = val_total_loss / len(validation_dataloader)            
      val_loss_values.append(avg_val_loss)

      # Report the final accuracy for this validation run.
      print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
      print("  Validation took: {:}".format(format_time(time.time() - t0)))
  return loss_values, val_loss_values
    

