### Zhuzhou dataset ###

gpu=0
cityname=zhuzhou

# model training
python train.py \
--cityname $cityname \
--gpu $gpu

# model testing
python test.py \
--cityname $cityname \
--gpu $gpu