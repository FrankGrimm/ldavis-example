```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Make sure to apply bugfix to `.venv/lib/python3.7/site-packages/pyLDAvis/utils.py`: https://github.com/bmabey/pyLDAvis/pull/158/commits/bc11c729a03b2c2903644d0cb738bd4c06b98057
```python
        if np.iscomplexobj(obj):
            return abs(obj)
```
