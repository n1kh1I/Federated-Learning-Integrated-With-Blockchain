import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import multiprocessing
import json
from web3 import Web3, HTTPProvider


# Replace 'http://localhost:8545' with the URL of your Ethereum node (e.g., Ganache)
web3 = Web3(HTTPProvider('https://fragrant-cosmological-silence.ethereum-sepolia.discover.quiknode.pro/8300eccc3e1913a290590a36219f3160b86c7c03/'))

# Check if the connection is successful
if web3.is_connected():
    print("Connected to Ethereum node:", web3.client_version)
else:
    print("Failed to connect to Ethereum node.")

# Replace 'YourContractABI' with the ABI of your smart contract
contract_abi = [
	{
		"inputs": [
			{
				"internalType": "bytes",
				"name": "modelData",
				"type": "bytes"
			}
		],
		"name": "storeModel",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "storedModelData",
		"outputs": [
			{
				"internalType": "bytes",
				"name": "",
				"type": "bytes"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]

# Replace 'YourContractAddress' with the address of your smart contract
contract_address = web3.to_checksum_address('0xAD8EC62372E065eF3EE496Fe04d5DeDAFfE6CCE9')

# Create a contract object
contract = web3.eth.contract(address=contract_address, abi=contract_abi)
# Loading the dataset from a text file


class DatasetPartition:
    def __init__(self, sentences):
        self.sentences = sentences
# Loading the dataset from a text file
def load_dataset(file_path):
    with open(file_path, "r") as f:
        sentences = f.read().splitlines()
    return sentences
# Data preprocessing
def preprocess_data(sentences):
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    return sequences, tokenizer, vocab_size
def read_sentences_from_file(file_path):
    sentences = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Remove leading/trailing whitespaces and add the sentence to the list
                sentences.append(line.strip())
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return sentences

def train_local_model(participant,max_sequence_length,tokenizer,vocab_size,model,global_model):
    participant_sequences = tokenizer.texts_to_sequences(participant.sentences)
    X = keras.preprocessing.sequence.pad_sequences(participant_sequences, maxlen=max_sequence_length)

    # Extract the last word index from each sequence
    y = np.array([seq[-1] for seq in participant_sequences])
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    if global_model is None:
        # Clone the model for the first participant
        local_model = keras.models.clone_model(model)
        local_model.compile(optimizer='adam', loss='categorical_crossentropy')
        local_model.set_weights(model.get_weights())
    else:
        # Use the global model for subsequent participants
        local_model = keras.models.clone_model(global_model)
        local_model.compile(optimizer='adam', loss='categorical_crossentropy')
        local_model.set_weights(global_model.get_weights())

    # Train the local model
    local_model.fit(X, y, epochs=1)

    # Append the weights of the local model to the list
    return local_model.get_weights()

# Parallel execution using multiprocessing.Pool
def parallel_execution(dataset, num_comm_rounds, num_selected_participants,max_sequence_length,tokenizer,vocab_size,model):
    global_model = None

    for comm_round in range(num_comm_rounds):
        # Randomly select participants for this round
        selected_participants = random.sample(dataset, num_selected_participants)

        args_for_participants = [(participant, max_sequence_length, tokenizer,vocab_size,model,global_model)
                                 for participant in selected_participants]

        # Use multiprocessing to train participants in parallel
        with multiprocessing.Pool(processes=num_selected_participants) as pool:
            local_weights_list = pool.starmap(train_local_model, args_for_participants)


        if global_model is None:
            global_model = keras.models.clone_model(model)
            global_model.compile(optimizer='adam', loss='categorical_crossentropy')
            global_model.set_weights(local_weights_list[-1])
        else:
            # Aggregate the local model weights by averaging
            global_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*local_weights_list)]
            global_model.set_weights(global_weights)
    
    return global_model

# Replace 'path/to/sentences.txt' with the actual path to your .txt file.

# Federated Learning Setup
'''def federated_learning(dataset, num_comm_rounds, num_selected_participants):
    

    for comm_round in range(num_comm_rounds):
        # Randomly select participants for this round
        selected_participants = random.sample(dataset, num_selected_participants)

        # Train on selected participants' data
        local_weights_list = []  # Store the model weights of selected participants

        for participant in selected_participants:
            ##### left over #####
        if global_model is None:
            global_model = local_model
        else:
            # Aggregate the local model weights by averaging
            global_weights = [np.mean(layer_weights, axis=0) for layer_weights in zip(*local_weights_list)]
            global_model.set_weights(global_weights)
    
    return global_model'''
if __name__ == '__main__':
    file_path = 'dataset/disease.txt'
    sentences = read_sentences_from_file(file_path)
    # print(sentences)

    num_sentences = 10000
    # dataset = [random.choice(sentences) for _ in range(num_sentences)]
    dataset = sentences

    # Preprocessing the dataset
    sequences,tokenizer, vocab_size = preprocess_data(dataset)

    # Spliting the dataset into input sequences and target words
    input_sequences = []
    target_words = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            target_words.append(sequence[i])
    # Padding the input sequences to have a fixed length
    global max_sequence_length
    max_sequence_length = max([len(sequence) for sequence in input_sequences])
    input_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_length)

    # Converting the target words to categorical one-hot encoding
    target_words = tf.keras.utils.to_categorical(target_words, num_classes=vocab_size)

    # Model architecture
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 10, input_length=max_sequence_length),
        keras.layers.LSTM(32),
        keras.layers.Dense(vocab_size, activation='softmax')
    ])
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generating the dataset partitions for participants
    dataset_partitions = [
        DatasetPartition(sentences[:5000]),  # Participant 1s
        DatasetPartition(sentences[5000:])   # Participant 2
    ]
    num_comm_rounds = 5
    num_selected_participants = 2

    trained_model = parallel_execution(dataset_partitions, num_comm_rounds, num_selected_participants, max_sequence_length, tokenizer,vocab_size,model)
    #save the trained model
    trained_model.save("trained_model.keras")
    # Generat ingpredictions
    input_sequence = ["Memory loss, disorientation, confusion"]
    insq = input_sequence
    print(insq)
    input_sequence = np.array(tokenizer.texts_to_sequences(input_sequence))[:, :-1]
    input_sequence = keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_sequence_length)
    predictions = trained_model.predict(input_sequence)
    predicted_word_index = np.argmax(predictions[0])
    predicted_word = tokenizer.index_word[predicted_word_index]

    # print(tokenizer.word_index)
    # original_input_sequence = tokenizer.sequences_to_texts([list(input_sequence[0])])[0]
    # input_sequence= input_sequence[0].split(',') 
    # Convert predictions back to words
    # predicted_word = tokenizer.sequences_to_texts([predictions])[0]
    predicted_word_indices = np.argmax(predictions, axis=1)
    predicted_word = tokenizer.sequences_to_texts([predicted_word_indices])[0]
    print("Predicted Disease:", predicted_word)
    insq = ['Memory loss, disorientation, confusion']
    split_insq = insq[0].split(', ')
    print(split_insq)
    # print(insq)
    name=input()
    # print(input_sequence)
    output_dict={'name':name, 'symptom 1': split_insq[0], 
    'symptom 2': split_insq[1],
    'symptom 3': split_insq[2],
    'predicted_disease': predicted_word}

    
    with open('output.json', 'w') as outfile:
        json.dump(output_dict, outfile)

    with open("output.json", 'rb') as f:
        model_data = f.read()

    # Encoding the binary data as hexadecimal
    hex_model_data = '0x' + model_data.hex()

    # Setting the gas price in GWei
    gas_price_wei = 100000000000

    private_key = 'df2ad361f8cb79f77076a9b34c309cda5e914e2b53a757219eb8b029fa7f3e6b'

    # Importing the account
    account = web3.eth.account.from_key(private_key)

    # Setting the gas limit
    transaction = contract.functions.storeModel(hex_model_data).build_transaction({
        'gas': 5000000,
        'gasPrice': gas_price_wei,
        'nonce': web3.eth.get_transaction_count(account.address),  # Use the account's address for the 'from' field
        'from': account.address,
    })
    # Signing the transaction
    signed_transaction = web3.eth.account.sign_transaction(transaction, private_key)

    # Sending the transaction
    tx_hash = web3.eth.send_raw_transaction(signed_transaction.rawTransaction)


    # Waiting for the transaction to be mined (optional)
    web3.eth.wait_for_transaction_receipt(tx_hash)

    # Displaying the transaction hash
    print("Transaction hash:", tx_hash.hex())