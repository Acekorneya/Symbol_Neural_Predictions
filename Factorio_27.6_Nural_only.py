import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
import pickle
import scipy.optimize
import os
import random
import tkinter as tk
from PIL import Image, ImageTk
import concurrent.futures
import matplotlib.pyplot as plt
import gc


initial_training_data = [
    (['t', 'aa', 'b', 'y', 'dd', 'n', 'qq', 'k'], [-0.96843255296311, 0.202010896323, -0.14604789669195]),
    (['iii', 'xx', 'uuu', 'cc', 'ss', 'kk', 'jj', 'ff'],[0.96896430287334, 0.20023642364133, -0.14496052705654]),
    (['t', 'w', 'oo', 'll', 'mm', 'f', 'p', 'o'], [-0.95413147037031, 0.21521484882988, -0.20812425637614]),
]


def save_to_file(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_from_file(filename, default_data=None):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        if default_data is not None:
            save_to_file(default_data, filename)
        return default_data

def save_to_backup(data, file_name, backup=False):
    if backup:
        backup_file_name = file_name[:-7] + "_backup.pickle"
        with open(backup_file_name, 'wb') as backup_file:
            pickle.dump(data, backup_file)

    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

symbol_vectors = load_from_file('C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\symbols_dict.pickle')
data  = load_from_file('C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\training_data.pickle',default_data=initial_training_data)

new_symbols = {
    '!': np.array([3]),
    '!!': np.array([3]),
    '!!!': np.array([3]),
    '!!!!': np.array([3]),
}
symbol_vectors.update(new_symbols)

# Add this function to display images corresponding to the symbols
def display_images(symbols, images_folder='C:\\Users\\aceko\\Desktop\\Dodeca\\Symbols'):
    window = tk.Tk()
    window.title("8 Best Symbols Connections")

    for symbol in symbols:
        image_path = os.path.join(images_folder, f'{symbol}.jpg')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((100, 100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(window, image=img)
            img_label.image = img
            img_label.pack(side=tk.LEFT)
        else:
            print(f"Image not found for symbol: {symbol}")

    window.mainloop()

def is_sequence_in_training_data(sequence, training_data):
    for symbols, _ in training_data:
        if symbols == sequence:
            return True
    return False

def find_optimized_symbols(training_data):
    best_symbols = None
    best_error = float("inf")
    num_samples = 10000

    while True:
        for _ in range(num_samples):
            sampled_symbols = random.choices(list(symbol_vectors.keys()), k=8)
            error = evaluate_symbols(sampled_symbols)

            if error < best_error:
                best_error = error
                best_symbols = sampled_symbols
        
        if not is_sequence_in_training_data(best_symbols, training_data):
            break

    return best_symbols

# Define a custom tokenizer function
max_sequence_length = 8  

# Define a custom tokenizer function
def custom_tokenizer(text, symbol_dict):
    tokens = []
    i = 0
    while i < len(text):
        for length in [4, 3, 2, 1]:
            if text[i:i+length] in symbol_dict:
                tokens.append(text[i:i+length])
                i += length
                break
            if length == 1:
                i += 1
    return tokens



X, y = zip(*data)
# Create a tokenizer
tokenizer = Tokenizer(char_level=True, filters="")
tokenizer.fit_on_texts(symbol_vectors.keys())

# Create a dictionary to map symbols to integer tokens
symbol_to_token = {symbol: idx+1 for idx, symbol in enumerate(symbol_vectors.keys())}

# Prepare the symbol embeddings matrix
vocab_size = len(symbol_to_token) + 1
embedding_dim = 3
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for symbol, idx in symbol_to_token.items():
    embedding_matrix[idx] = symbol_vectors[symbol]

save_dir = "C:\\Users\\aceko\\Desktop\\Dodeca\\Factorio_AI_folds"

def train_models():
    # Implement K-fold cross-validation, e.g., 5-fold cross-validation
  kfold = KFold(n_splits=5, shuffle=True, random_state=42)
  fold = 0

  models = []
  val_losses = []

  for train_index, val_index in kfold.split(X):
      fold += 1
      print(f"Training fold {fold}")
      X_train, X_val = [X[i] for i in train_index], [X[i] for i in val_index]
      y_train, y_val = [y[i] for i in train_index], [y[i] for i in val_index]

      # Tokenize the input sequences using the custom tokenizer
      X_train_tokens = [custom_tokenizer(''.join(seq), symbol_vectors) for seq in X_train]
      X_val_tokens = [custom_tokenizer(''.join(seq), symbol_vectors) for seq in X_val]

      # Convert the tokenized sequences to integer tokens using the mapping
      X_train_token_ids = [[symbol_to_token[symbol] for symbol in seq] for seq in X_train_tokens]
      X_val_token_ids = [[symbol_to_token[symbol] for symbol in seq] for seq in X_val_tokens]

      # Pad the sequences to a fixed length
      max_sequence_length = 8
      X_train_padded = pad_sequences(X_train_token_ids, maxlen=max_sequence_length, padding="post")
      X_val_padded = pad_sequences(X_val_token_ids, maxlen=max_sequence_length, padding="post")

      # Create the model
      input_layer = Input(shape=(max_sequence_length,))
      embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)(input_layer)
      lstm_layer1 = LSTM(256, return_sequences=True)(embedding_layer)
      dropout_layer1 = Dropout(0.5)(lstm_layer1)
      lstm_layer2 = LSTM(128, return_sequences=True)(dropout_layer1)
      dropout_layer2 = Dropout(0.5)(lstm_layer2)
      lstm_layer3 = LSTM(64)(dropout_layer2)
      output_layer = Dense(3)(lstm_layer3)

      model = Model(inputs=input_layer, outputs=output_layer)
      

      # Compile the model
      learning_rate = 0.0001
      optimizer = Adam(learning_rate=learning_rate)
      
      model.compile(optimizer=optimizer, loss="mse")

      model.summary()
      
      # Create an early stopping callback
      early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

      # Train the model
      history = model.fit(
          X_train_padded,
          np.array(y_train),
          epochs=1500,
          batch_size=32,
          validation_data=(X_val_padded, np.array(y_val)),
          verbose=1,
          callbacks=[early_stopping]
      )
      # Evaluate the model's performance
      model.evaluate(X_val_padded, np.array(y_val))

      # Save the model for each fold
      model.save(os.path.join(save_dir, f"model_fold_{fold}.h5"))

      # Append the model and its validation loss to the lists
      models.append(model)
      val_losses.append(min(history.history['val_loss']))
      

  return models, val_losses

models, val_losses = train_models()

# Load the saved histories and calculate the average
histories = []
for fold in range(1, 6):
    with open(os.path.join(save_dir, f"history_fold_{fold}.pickle"), "rb") as file:
        histories.append(pickle.load(file))

# Calculate average of the training losses and validation losses
avg_train_loss = np.mean([hist["loss"] for hist in histories], axis=0)
avg_val_loss = np.mean([hist["val_loss"] for hist in histories], axis=0)

# Plot the average training and validation losses
plt.plot(avg_train_loss, label="Train Loss")
plt.plot(avg_val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

target_vector = np.array([0.96610584537205, 0.060309829005427, -0.25100243039319])
models = []
val_losses = []

for fold in range(1, 6):
    model_path = os.path.join(save_dir, f"model_fold_{fold}.h5")
    loaded_model = load_model(model_path)
    models.append(loaded_model)

    history_path = os.path.join(save_dir, f"history_fold_{fold}.pickle")
    with open(history_path, "rb") as file:
        history = pickle.load(file)
        val_losses.append(min(history["val_loss"]))

# Find the best model with the best validation loss
best_model_index = np.argmin(val_losses)
best_model = models[best_model_index]

# Add the following line to print the best model's index
print(f"Best model i'll be using: model_fold_{best_model_index + 1}.h5")

def check_existing_combination(symbols, data):
    for entry in data:
        if set(entry[0]) == set(symbols):
            return True
    return False
  
def evaluate_symbols(symbols):
    token_ids = [symbol_to_token[symbol] for symbol in symbols]
    sequence_padded = pad_sequences([token_ids], maxlen=max_sequence_length, padding="post")
    output = best_model.predict(sequence_padded)
    return np.linalg.norm(output - target_vector)

def find_optimized_symbols(training_data):
    best_symbols = None
    best_error = float("inf")
    num_samples = 15000

    def evaluate_sample(_):
        #random.sample will generate a list of unique items (i.e., without replacement) from the input list, ensuring that the same symbol does not appear multiple times
        sampled_symbols = random.sample(list(symbol_vectors.keys()), k=8)
        #random choice can pick same symbol on any slot
        #sampled_symbols = random.choices(list(symbol_vectors.keys()), k=8)
        error = evaluate_symbols(sampled_symbols)
        return error, sampled_symbols

    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for error, symbols in executor.map(evaluate_sample, range(num_samples)):
                if error < best_error:
                    best_error = error
                    best_symbols = symbols
        
        if not is_sequence_in_training_data(best_symbols, training_data):
            break

    # Print the model's predicted vector for the best symbol sequence
    best_symbols_token_ids = [symbol_to_token[symbol] for symbol in best_symbols]
    best_symbols_sequence_padded = pad_sequences([best_symbols_token_ids], maxlen=max_sequence_length, padding="post")
    best_predicted_vector = best_model.predict(best_symbols_sequence_padded)

    return best_symbols, best_predicted_vector.ravel()

first_run = True
while True:
    if first_run:
        best_symbols, predicted_vector = find_optimized_symbols(data)
        print(f"Model predicted vector for the best symbol sequence: {predicted_vector}")
        print(f"Optimized symbol combination: {best_symbols}")
        if not check_existing_combination(best_symbols, data):
            display_images(best_symbols)
        else:
            print("The optimized symbol combination already exists in the training data.")

    first_run = False

    action = input("Do you want to rerun the model, view the training data, add training data, edit data, or exit? (rerun/view/add/edit/exit) ")

    if action.lower() == 'rerun':
        rerun_option = input("Do you want to (1) find other symbol sequences or (2) retrain the model? (1/2) ")

        if rerun_option == '1':
            gc.collect()
            best_symbols, predicted_vector = find_optimized_symbols(data)
            print(f"Model predicted vector for the best symbol sequence: {predicted_vector}")
            print(f"Optimized symbol combination: {best_symbols}")
            display_images(best_symbols)

        elif rerun_option == '2':
            gc.collect()
            retrain = input("Do you want to retrain the models? (yes/no) ")
            if retrain.lower() == 'yes':
                models, val_losses = train_models()
            best_symbols, predicted_vector = find_optimized_symbols(data)
            print(f"Model predicted vector for the best symbol sequence: {predicted_vector}")
            print(f"Optimized symbol combination: {best_symbols}")
            display_images(best_symbols)
        else:
            print("Invalid option. Please enter '1' or '2'.")
        
    elif action.lower() == 'view':
        sorted_data = sorted(data, key=lambda x: np.linalg.norm(np.array(x[1]) - target_vector))
        print(sorted_data)

    elif action.lower() == 'add':
        custom_or_predicted = input("Do you want to enter a custom set of symbols or the predicted symbols? (custom/predicted) ")

        if custom_or_predicted.lower() == 'custom':
            custom_symbols = input("Enter the custom symbol sequence separated by commas: ")
            best_symbols = [s.strip().replace("'", "") for s in custom_symbols.split(',')]

        game_vector_input = input("Enter the actual game vector data (comma-separated values) or leave it blank to go back: ")

        if not game_vector_input:
            continue

        try:
            game_vector = np.array([float(x) for x in game_vector_input.split(',')])
        except ValueError:
            print("Invalid input. Please enter game vector data in the correct format.")
            continue

        error = np.linalg.norm(game_vector - predicted_vector)
        print(f"Error between predicted and actual game vectors: {error}")

        save_training_data = input("Do you want to save the symbols and game vector data to the training data? (yes/no) ")

        if save_training_data.lower() == 'yes':
            data.append((best_symbols, game_vector.tolist()))
            save_to_file(data, 'C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\training_data.pickle')

            create_backup = input("Do you want to create a backup of the training data? (yes/no) ")

            if create_backup.lower() == 'yes':
                save_to_backup(data, 'C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\training_data.pickle', backup=True)

    elif action.lower() == 'edit':
        view_data = input("Do you want to view the training data? (yes/no) ")
        
        if view_data.lower() == 'yes':
            sorted_data = sorted(data, key=lambda x: np.linalg.norm(np.array(x[1]) - target_vector))
            for i, entry in enumerate(sorted_data):
                print(f"{i}: {entry}")

        delete_data = input("Enter the index number of the data you want to delete or leave it blank to go back: ")

        if not delete_data:
            continue

        try:
            index = int(delete_data)
            if 0 <= index < len(data):
                del data[index]
                print("Deleted the selected data.")
            else:
                print("Invalid index. Please enter a valid index number.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid index number.")
            continue

        save_edited_data = input("Do you want to save the edited training data? (yes/no) ")
        
        if save_edited_data.lower() == 'yes':
            save_to_file(data, 'C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\training_data.pickle')

            create_backup = input("Do you want to create a backup of the training data? (yes/no) ")

            if create_backup.lower() == 'yes':
                save_to_backup(data, 'C:\\Users\\aceko\\Desktop\\Dodeca\\Neural_dic\\training_data.pickle', backup=True)

    elif action.lower() == 'exit':
        break

    else:
        print("Invalid action. Please enter 'rerun', 'view', 'add', 'edit', or 'exit'.")