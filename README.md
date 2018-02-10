# active-manifesto
An Active Learning approach for political text annotation using the manifesto project corpus. 

The experiments are explained in this [manuscript](https://github.com/felixbiessmann/active-manifesto/blob/master/manuscript/active-manifesto.pdf).

A demo using active learning to annotate texts that uses some annotations to estimate the political bias of annotators to recommend 'unbiasing' news can be found [here](http://www.rightornot.info/)

## Running experiments
Create virtualenv:

  `python3.6 -m venv venv`

Activate virtualenv:

  `source venv/bin/activate`

Install dependencies:

  `pip install -r requirements.txt`

Add API-Key to `manifesto_data.py` ([see Manifesto Project Website](https://manifestoproject.wzb.eu/information/documents/api)).

## Deployment of services

See Readme.md in `services` directory.
