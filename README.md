# Next Word Prediction

Simple Next word prediction model using an LSTM. You can provide a text file to train on, or specify a wikipedia article you want to train the model on. 

## Examples 

Train the model on an input text file, then predict the next word in the phrase "Travelling day in and day": 

`python model.py --train_model --input_text meta_clean.txt --epochs 20 --predict "Travelling day in and day"`

Train the model on the wikipedia article for "Basketball", and predict the next word in the phrase "Basketball is a "

`python model.py --train_model --wiki "Basketball"  --epochs 100 --predict "Basketball is a "`

If you have a pretrained model, you can load the model and predict the next word like so:

`python model.py --predict "The NBA was started in "`
