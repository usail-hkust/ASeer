### Baoding dataset ###

gpu=0
cityname=baoding

# model training
python train.py \
--cityname $cityname \
--gpu $gpu

# model testing
python test.py \
--cityname $cityname \
--gpu $gpu
