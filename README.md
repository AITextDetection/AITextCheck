### Run the api 

```uvicorn app:app --host 0.0.0.0 --port 8000```

`python -m src.tokenize_data`
`source venv/bin/activate`

`pip install -r requirements.txt`
`pip freeze > requirements.txt`

`tensorboard --logdir=./logs`


## Final Evaluation 

`python src/evaluate_model.py --data path/to/your/test.csv --batch_size 32`

`python src/evaluate_model.py`

`python main.py --train`