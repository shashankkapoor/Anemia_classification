# A Simple Keras + deep learning REST API

This repository contains the code for [*Building a simple Keras + deep learning REST API*].

The method covered here is intended to be instructional. It is _not_ meant to be production-level and capable of scaling to heavy load.

## Pre-requisite
```sh
$ pip install -r requirments.txt
```

## Getting started

I assume you already have Keras (and a supported backend) installed on your system. From there you need to install [Flask](http://flask.pocoo.org/) and [requests](http://docs.python-requests.org/en/master/):

```sh
$ pip install flask gevent requests
```

Next, clone the repo:

```sh
$ git clone https://github.com/shashankkapoor/Anemia_classification.git
```

## Starting the Keras server

Below you can see the image we wish to classify, Anemic or Non-Anemic :


The Flask + Keras server can be started by running:

```sh
$ python demo.py 
Using TensorFlow backend.
 * Loading Keras model and Flask starting server...please wait until server has fully started
...
 * Running on http://127.0.0.1:5000
```

You can now access the REST API via `http://127.0.0.1:5000`.

## Submitting requests to the Keras server

Requests can be submitted via cURL:

```sh

$curl -X POST -F image=@'image/palm_image/Anemic-264 (11).png' 'http://127.0.0.1:5000/predict'

{
  "predictions": [
    {
      "label": "Anemic", 
      "probability": 0.9901360869407654
    }, 
    {
      "label": "Non-Anemic", 
      "probability": 0.002396771451458335
    }
  ], 
  "success": true
}
```

