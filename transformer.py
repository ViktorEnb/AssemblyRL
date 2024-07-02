import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
import torch.optim as optim
import numpy as np
import assembly

def base_repr_to_int(base_repr_str, base):
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = 0
    negative = False
    
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36, inclusive.")
    
    # Check for negative sign
    if base_repr_str.startswith('-'):
        negative = True
        base_repr_str = base_repr_str[1:]
    
    for char in base_repr_str:
        value = digits.index(char)
        if value >= base:
            raise ValueError(f"Invalid character '{char}' for base {base}.")
        result = result * base + value
    
    if negative:
        result = -result
    
    return result


class AssemblyInstructionsRepresentationModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(AssemblyInstructionsRepresentationModel, self).__init__()
        self.instructions = ["MOV", "ADD", "CMP", "R1", "R2", "R3", "R4"]
        self.vocab_size = len(self.instructions) ** 3 #TODO: make this real
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_layer = nn.Linear(self.vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, self.vocab_size)   

    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.output_layer(x)
        softmax = nn.Softmax(dim=1)
        x = softmax(x)
        return x
    
    def onehot_decode_line(self, onehot : torch.tensor):
        value = (onehot == 1).nonzero(as_tuple=False)[0]
        base_repr = np.base_repr(value, len(self.instructions))
        base_repr = "0" * (3 - len(base_repr)) + base_repr
        return [self.instructions[int(c)] for c in base_repr]

    def onehot_encode_line(self, line):
        words = line.split(" ")
        counter = len(words) - 1
        base = len(self.instructions)
        value = 0
        for word in words:
            value += base ** counter * self.instructions.index(word)
            counter -= 1
        out = F.one_hot(torch.LongTensor([value]), num_classes = self.vocab_size)
        out = out.float()
        return out
    def onehot_encode_program(self, program):
        lines = program.split("\n")
        program_repr = []
        for line in lines:
            words = line.split(" ")
            encoded_line = torch.tensor([[0 + (instruction == word) for instruction in self.instructions] for word in words])
        program_repr.append(encoded_line)

# Define a simple transformer-based model
class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(SimpleTransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

#Train the SimpleTransformerModel on learn basic task to test basic functionality
def test_train():
    model = SimpleTransformerModel(input_dim=5, hidden_dim=5, num_heads=1, num_layers=4)
    
    # Generate random training data
    train_data = torch.rand(200, 1, 5)  * 10 # Reshape to (batch_size, seq_length, input_dim)
    train_labels = 2 * train_data  # Targets are the input multiplied by 2

    test_data = torch.rand(20, 1, 5) * 10# Reshape to (batch_size, seq_length, input_dim)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Experiment with learning rates
    criterion = nn.MSELoss()  # Mean Squared Error Loss

    # Training loop
    for epoch in range(1000):  # Train for more epochs
        for x, y in zip(train_data, train_labels):
            optimizer.zero_grad()
            predicted_value = model(x)
            loss = criterion(predicted_value, y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Print loss every 50 epochs to monitor training
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    # Generate random test data
    with torch.no_grad():
        for x in test_data:
            print("x: ", x, "   model(x): ", model(x))    


# Example usage
if __name__ == "__main__":
    # test_train()
    # model = AssemblyInstructionsRepresentationModel(d_model = 64, num_heads = 1, num_layers=2)
    # onehot = model.onehot_encode_line("R4 R2 ADD") 
    # output_probabilities = model(onehot) 
    # best_line = F.one_hot(torch.LongTensor(torch.argmax(output_probabilities)), num_classes = model.vocab_size)
    # best_line = best_line.float()
    # print(best_line)
    # print(model.onehot_decode_line(best_line))
    a = Assembly()
    