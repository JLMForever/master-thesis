import yaml
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense

# 1. Reading YAML Files
with open('training.yml', 'r', encoding='utf-8') as f:
    train_entries = yaml.safe_load(f)   # List[Dict]

# 2. Extract all training sentences and their labels
texts = []
labels_domains    = []
labels_subdomains = []
labels_phases     = []
labels_dimensions = []
labels_level = []
labels_objectives = []

for entry in train_entries:
    ds = entry['domains'][0]
    sd = entry['subdomains'][0]
    ph = entry['phases'][0]
    di = entry['dimensions'][0]
    lv = entry['level'][0]
    ob = entry['objectives'][0]
    for txt in entry['input']:
        texts.append(txt)
        labels_domains.append(ds)
        labels_subdomains.append(sd)
        labels_phases.append(ph)
        labels_dimensions.append(di)
        labels_level.append(lv)
        labels_objectives.append(ob)

# 3. Construct a mapping for each tag(label → index)
domains    = sorted(set(labels_domains))
subdomains = sorted(set(labels_subdomains))
phases     = sorted(set(labels_phases))
dimensions = sorted(set(labels_dimensions))
level      = sorted(set(labels_level))
objectives = sorted(set(labels_objectives))

label_maps = {
    'domains':    {v:i for i,v in enumerate(domains)},
    'subdomains': {v:i for i,v in enumerate(subdomains)},
    'phases':     {v:i for i,v in enumerate(phases)},
    'dimensions': {v:i for i,v in enumerate(dimensions)},
    'level':      {v:i for i,v in enumerate(level)},
    'objectives': {v:i for i,v in enumerate(objectives)},
}

# 4. Text Serialization & labels One-Hot
# Tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
seqs = tokenizer.texts_to_sequences(texts)
MAX_LEN = 50
X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
vocab_size = len(tokenizer.word_index) + 1

# One-hot encoding function
def make_onehot(label_list, mapping):
    arr = np.zeros((len(label_list), len(mapping)), dtype='float32')
    for i, lbl in enumerate(label_list):
        arr[i, mapping[lbl]] = 1.0
    return arr

y_domains    = make_onehot(labels_domains,    label_maps['domains'])
y_subdomains = make_onehot(labels_subdomains, label_maps['subdomains'])
y_phases     = make_onehot(labels_phases,     label_maps['phases'])
y_dimensions = make_onehot(labels_dimensions, label_maps['dimensions'])
y_level      = make_onehot(labels_level, label_maps['level'])
y_objectives = make_onehot(labels_objectives, label_maps['objectives'])

# 5. Building the LSTM model
embed_dim = 64
lstm_units = 64

inp = Input(shape=(MAX_LEN,), name='text_input')
x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(inp)
x = LSTM(lstm_units)(x)

out_domains    = Dense(len(domains),    activation='softmax', name='domains')(x)
out_subdomains = Dense(len(subdomains), activation='softmax', name='subdomains')(x)
out_phases     = Dense(len(phases),     activation='softmax', name='phases')(x)
out_dimensions = Dense(len(dimensions), activation='softmax', name='dimensions')(x)
out_level      = Dense(len(level), activation='softmax', name='level')(x)
out_objectives = Dense(len(objectives), activation='softmax', name='objectives')(x)

model = Model(inputs=inp,
              outputs=[out_domains, out_subdomains, out_phases, out_dimensions, out_level, out_objectives])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 6. Training model
model.fit(
    X,
    [y_domains, y_subdomains, y_phases, y_dimensions, y_level, y_objectives],
    epochs=100,
    batch_size=8
)

# 7. Save the model and artifacts
model.save('reflection_model.h5')

# Save tokenizer and label maps
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_maps.pkl', 'wb') as handle:
    pickle.dump(label_maps, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save MAX_LEN
with open('config.pkl', 'wb') as handle:
    pickle.dump({'MAX_LEN': MAX_LEN}, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and artifacts saved successfully!")