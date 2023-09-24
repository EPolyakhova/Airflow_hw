import json
import dill
import os
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')

def predict():
    model_list = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{model_list[-1]}', 'rb') as file:
        model = dill.load(file)

    pred_files = os.listdir(f'{path}/data/test')
    id_list = []
    pred_list = []

    for json_file in pred_files:
        with open(f'{path}/data/test/{json_file}') as file:
            form = json.load(file)
        df = pd.DataFrame.from_dict([form])
        prediction = model.predict(df)
        id_list.append(df.loc[0,'id'])
        pred_list.append(prediction[0])

    preds_data = {'car_id': id_list, 'pred': pred_list}
    df = pd.DataFrame(preds_data)
    df.set_index('car_id', inplace=True)
    df.to_csv(f'{path}/data/predictions/predict_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
