
# MLOPS Project Description - NLP Sentiment Analysis of iMDB reviews

s224959 Sofus Carstens <br />
s204701 Lukas Raad <br />
s225083 Christian Warburg <br />

## How to run our project

Step 1: Create new virtual enviroment using `conda env create -f env.yml` <br />
Step 2: Get requirements from `requirements.txt` file <br />
Step 3: Configure experiment in the config files <br />
Step 4: Run `src/make_dataset.py` to download and create the imdb dataset <br />
Step 5: Run `src/train.py` to train the model <br />
Step 6: Run the `src/predict.py` to run inference/test the model <br />

## Overall goal for the project

The goal of this project is to solve a binary classification task whether a review is negative or positve.

## Framework for our project

To solve this classification task we plan to use the hugging face Transformers repo. More specifically a pretrained model with a classification head (BERT)

## Dataset (might change)

We are using the IMDb dataset from Hugging Face. Each sample in the dataset consists of a movie review along with a binary sentiment label indicating whether the review is positive (1) or negative (0). The dataset is widely used for sentiment analysis tasks and includes reviews of varying lengths and complexities. The dataset was chosen because it provides a balanced and comprehensive set of text samples for exploring Natural Language Processing techniques. Its straightforward binary classification task makes it well-suited for developing and evaluating machine learning models in a limited timeframe.

## What deep learning models do you expect to use?
We plan to use **DistilBERT**, a lightweight transformer-based model designed for natural language understanding tasks. DistilBERT is a smaller, faster, and more efficient version of BERT, making it an ideal choice for our sentiment analysis task on the IMDb dataset. Its reduced size allows for quicker training and inference without significant loss in performance, making it feasible to implement within our timeframe while still leveraging the power of pre-trained transformer models.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [ ] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [X] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [ ] Get some continuous integration running on the GitHub repository (M17)
* [ ] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

67

### Question 2
> **Enter the study number for each member in the group**
>
>
>
>
>
> Answer:

s204701, s225083, sXXXXXX

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
> Answer:

To accurately classify the IMDB movie reviews, we used the Transformers framework by
huggingface. This framework allowed us to use a pretrained model which we could finetune,
and also had an autotokenizer function which allowed us to effectively convert the reviews
to tokens. We experimented with some of the models, but ended up using the tiny-bert
model, because of its small size that could be trained quickly and it still had good results.
It helped us complete project because it had valueable functions such as auto-tokenizer and
pre-trained models such BERT for classification. This simplies the project giving us more
time to focus on the MLOps part of the course.


## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Answer:

We had all used conda before so we decided to use conda to manage our dependencies in the projet. We created a requirements file first with a simple `pip freeze \> requirements.txt`. However we realized that this created a rather large list that was quite difficult to quickly get an idea of the packages and versions required. Therefore, we changed to `pipreqs` which provides a far briefer list. We also made use of the built-in conda commands to export environments to .yml. When a new team member wanted to get an exact copy of our environment he could run `conda env create -f env.yml` to create the environment on their local machine. This could also be done using the command 'uv venv', which looks at the 'pyproject.toml' file and copies the dependencies.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Answer:

We deviated from the template by removing the subfolder from the 'src' folder. We did this so it would look cleaner.
Otherwise we thought that the template matched accurately with our project, and decided to keep it as it were for the most part.
Of all the folders under the cookie cutter template, many were left empty. We mainly filled out

* the src folder with all the model, train, prediction and data code
* the tests folder with the defintions for our pytests
* the configs with the config files we used for configuring everything
* the data folder is there but we don't push any data to it since the files are too large for github

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

Yes, we used flake8 to highlight missing packages and unused variables to make sure to catch
any errors in the early stages. Furthermore, we made sure to write meaningful comments
in the codebase to emphasize and explain the code. We also made sure to keep the naming
formatting consistent and clean, i.e keeping variables with underscores and lower lettering.
Everytime to committed changes to the code we tried to write meaningfull commit mes-
sages to ensure that everybody in the project know what had changed since last time.in
larger projects, writing thorough documentation is extremely important so that there are no
ambiguities on how the code is implemented and people can quickly get up to speed.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented tests for the data and for the model. The test for the data ensures that the
data exists, has the correct format and the correct size. The test for the model ensures that
the model is found and loaded successfully, and that the output shape is correct i.e. binary
since we’re dealing with classification.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

With the unit tests we had implemented locally, we managed to achieve a code coverage of
77 percent. If our code did have a coverage close to 100% we would still not trust it to be
error free, because code coverage does not guarantee that there are no bugs. Code coverage
simply informs us on what percentage of our code gets run every time we run our tests. That
means there can be some small cases that aren’t covered by our tests and would still result
in a bug in certain situations. On the other hand, a high code coverage would still give us a
good idea of how well our code is running although we cannot take it as face value.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used branches and pull requests in our workflow. Branching was helpful in allowing
each team member to work on features by themselves to ensure that our main codebase
remained stable. We typically created a new branch whenever we started work on a new
feature or bug fix. Once the feature was completed and tested locally, we would merge the
branch back into the main using a pull request. This will allow to verify if there was any
merge conflict or issues. This process made it possible to roll back changes if necessary,
without disrupting other parts of the codebase.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not use DVC for managing data in our project. However, having version control
for data can be very beneficial in situations where the dataset changes frequently. For
example, if a new dataset version is released or if more samples are collected over time,
version-controlling the data would allow us to trace how performance changes with each
dataset update. Also, it would enable us to quickly revert to a previous dataset version if a
newly introduced change or corruption unexpectedly makes the model perform worse.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We made use of unittesting, linting and github actions for our code. We started by
writing the unittest’s, and later implementing the github actions to make sure that every
push to github is passing those tests. We also used flake8 to make sure our code stays ”clean”
and to catch common pitfalls early. By running flake8 as part of our development routine
, we maintain consistent formatting, avoid naming clashes, and prevent unused imports or
variables from lingering in our repository.
The unittesting was also something that we implemented to catch errors in the process
of coding the project. This made sure that the data was on the correct format and that the
model was applicable. Later, we used github actions which essentially runs our tests each
time somebody pushes any changes to github.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used YAML configuration files alongside Hydra to define all our experiment parame-
ters such as learning rate, batch size, model architecture (We experimented with multiple
BERT models) at the start of each run. This approach let us cleanly separate code from
configuration and made it easy to tweak hyperparameters without rewriting scripts.
### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To secure that no information get lost, we used hydra’s configuration file logging system.
In that way, in the output of each run, we had the hydra configuration files logged so that
whenever someone wanted to check a result, he could open the hydra config files that were
used to produce a specific output model and then he could get the hyperparameters used for
the training. However, we know that results in machine learning are largely related to the
data used in training, which is why applying version control to our data is also a fundamental
step of reproducibility in machine learning environments.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

• train/learningrate: How the learning rate changes over the course of training. Some
runs use a scheduler that gradually lowers the learning rate, leading to different final
values.
• train/epoch: Tracks which epoch the model is in as training steps increase. It helps
confirm that all runs complete the intended number of epochs.
• train/loss: Shows how the loss decreases (generally) with more steps—lower loss indi-
cates the model is learning effectively. A sharp drop at the beginning often means the
model quickly adapts to the dataset.
• Simply how many gradient-update steps have been performed. Useful for comparing
runs with different batch sizes or epoch configurations.
• train/gradnorm:Simply how many gradient-update steps have been performed. Useful
for comparing runs with different batch sizes or epoch configurations.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we built a docker file for our training file (train.py) using "docker build -f dockerfiles/train.dockerfile . -t train:latest".
Where the train.dockerfile defines what we need to RUN/COPY and pip install for the train.py to work. We then tested the docker image in a docker container locally
using the "docker desktop" app to make sure it worked.

For our project we built a docker file for our training file (train.py) using:
docker build -f dockerfiles/train.dockerfile . -t train:latest
Where the train.dockerfile defines what we need to RUN/COPY and pip install for
train.py to work on any pc. We then tested the docker image in a docker container locally
using the Docker Desktop app to make sure it worked correctly. We ofcourse tested the
dockerfiles on multiple pc’s and it worked, which is the point of dockerfiles.
### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Usually, we use the VS Code debugger tool in order to debug our code. This allows us to
step through the code line by line after specifying a line to stop at. Then we can also make
use of the debug console to check the datatypes, sizes, etc. of variables. Unfortunately, we
did not reach to perform any profiling on our code; however, given the time we would have
started by profiling the data loading files because that is where we feel the highest bottleneck
was.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We mainly made use/tried to make use of 3 GCP services, Buckets, compute engine and
artifact registry. We used buckets to store things like our raw data, preprocessed data,
model and api python scripts. The bucket just makes it easy for managing large data and
easily sharing it in the team/group. We tried using the Compute Engine and successfully
spun up a virtual machine with our environment, but making a training job was very difficult
(and at times frustrating). One issue was matching CUDA and PyTorch versions with the
VM’s GPU drivers

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

We didnt use the GCP cloud build history

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We did not manage to train our model in the cloud using either the Compute Engine
or Vertex AI. Initially, we attempted using the Compute Engine by spinning up a virtual
machine with the necessary environment (Docker, CUDA drivers, etc.) and transferring
our code to it. However, due to time constraints and some configuration hurdles—like
setting up the correct machine images, installing compatible GPU drivers, and ensuring our
Python dependencies worked seamlessly—we couldn’t finalize the training process before
the deadline. It would have been smart to use the compute engine because it allows to use
powerful GPUs, which is basically a must for training good models.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

In our api.py file we implemented an API for our prediction model with FastAPI. We
use two endpoints: Firstly a root endpoint (@app.get) just to make sure the api is running
and works. Mainly we include a /predict (@app.post to handle data submission) endpoint,
which gets our predictions from the model. In the ”/predict” we load our model alongside
the pretrained tokenizer. When we then send a request, in form of a text, to the /predict
endpoint our api encodes the text into token IDs and attention masks. We make these into
tensors (pytorch) and then pass them through our model to get a prediction (1 or 0).

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we managed to deploy our API locally. We used the FastAPI framework to achieve
this. In that way we access the FastAPI interface on a localhost website. The code was very
simple, we simply needed to load the deployable model from .pt file and set it in evaluation
mode. Then we encode the text submitted to the API and pass it through our model and
return the prediction of the model (1 if it’s positive or 0 if negative).

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

Unfortunately, we did not reach to perform any unit testing on our API. There are two main
types of test to perform: functionality and load. To test functionality, we can create a new
client to send requests which test specific applications in our API. Then we can write tests
using assert statements on the response from the API. For load testing we can use a service
like locust. This will simulate several users to observe how the API deals with several users.
However, for a local deployment, this doesn’t make a lot of sense because there won’t be
many users at a time. However, for an API that is deployed on the cloud that several people
can use at the same time, this is very important to test for.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to deploy monitoring. However, monitoring would help the longevity
of our model because it allows us to keep track of the performance of our deployed model.
We can both monitor classic DevOps performance so checking for errors in the operation
of the applications, or ensuring that logs are being updated correctly so we know what is
happening at a given moment, and monitoring how fast our application is working. In the
case of MLOps, we would also monitor data drifting with the evidently framework. This
occurs because if the model receives data that was outside its training scope, the distribution
of a feature in the data will increase over time. Then using evidently, we can test for data
drift and then perpetually retrain the model over time to ensure that the performance doesn’t
degrade over time.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We ended up spending 7.22 dollars in credit. We tried a bunch of different CPU’s and
GPU’s which ended up being the most expensive. Bucket storage was not very expensive
only 0.026 per Gigabyte per month. We would say that working in the cloud seems very
beneficial and very important, but it was also very difficult to learn because there is so much
information.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The overall struggles of this project was mainly in the department
of the cloud. It was a completely new thing for us, because we’ve only used the HPC on dtu
and our local machines for training models. This was of course a fun thing to interact with,
but there is so much information on the webpage of the cloud that we probably would need
a bit more time to completely understand everything that’s going on with all the services
to pick from. We tried to make use of the compute engine, but unfortunately we did not
manage to train the model in the cloud. We also had some trouble with the last commit
regarding the readme.md file of the project which did not go through and caused us a lot
of stress 2 weeks after submission. We also had some trouble with pytest which was also
completely new to us, but we managed to fix it after some research and trying different
things.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

We all contributed equally to the project.
