![](https://imgur.com/h0Dkvlb.jpg)

DeepLearning and Litics Analytics (Placeholder README, currently under review)
=======================

The application is called Litics Analytics, it is intended to be used with a range of different datasets to give the user a visual interpretation of said data. Statistical analysis is a major part of Machine Learning, and this generally involves using very large datasets that are very difficult to visualise. Cleaning this data is a major factor in training a successful model, which is usually easier whenever you can actually see the anomalous data input or output. 
This is what Litics does, it provides a platform for non-specific data visualisation and a basis for further development. 


## Getting Started

These instructions will get you up and running in no time!

### Prerequisites

System Software

```
Docker https://www.docker.com/get-started
```

```
Node.js https://nodejs.org/en/
```

```
Google Chrome or Firefox
```

### Installing

The easiest way to get started

```
#Get the project
Git Clone the project/Unzip

#Change directory
cd myproject

#Install NPM Dependencies
npm install

#Build Docker Image (this may take 3 minutes)
docker-compose build web

#Start Docker Image and server
docker-compose up web

```
#Linux Instructions
#Start docker if you need to (linux)
sudo service docker start
sudo docker-compose build web
sudo docker-compose up web


```
Thats it!
The docker script included handles starting ngrock and mongo with the server on Localhost:8080
```


## First time

The first run may be a little slow to start up, as the image needs built

### Create an account

The server won't be populated with test accounts, create one via Facebook, Github or LinkedIn or locally with email. 

```
Select Create account in the application header
```


## Built With

    "@fortawesome/fontawesome-free": "^5.11.2",
    "@ladjs/bootstrap-social": "^7.0.2",
    "@octokit/rest": "^16.32.0",
    "axios": "^0.19.0",
    "bcrypt": "^3.0.6",
    "body-parser": "^1.19.0",
    "bootstrap": "^4.3.1",
    "chalk": "^2.4.2",
    "chart.js": "^2.8.0",
    "compression": "^1.7.4",
    "connect-mongo": "^3.0.0",
    "dotenv": "^8.1.0",
    "errorhandler": "^1.5.1",
    "express": "^4.17.1",
    "express-flash": "^0.0.2",
    "express-handlebars": "^3.1.0",
    "express-session": "^1.16.2",
    "express-status-monitor": "^1.2.7",
    "fbgraph": "^1.4.4",
    "jquery": "^3.4.1",
    "lodash": "^4.17.15",
    "lusca": "^1.6.1",
    "mailchecker": "^3.2.35",
    "moment": "^2.24.0",
    "mongoose": "^5.7.3",
    "morgan": "^1.9.1",
    "multer": "^1.4.2",
    "node-sass": "^4.12.0",
    "node-sass-middleware": "^0.11.0",
    "nodemailer": "^6.3.0",
    "passport": "^0.4.0",
    "passport-facebook": "^3.0.0",
    "passport-github": "^1.1.0",
    "passport-linkedin-oauth2": "^2.0.0",
    "passport-local": "^1.0.0",
    "passport-oauth": "^1.0.0",
    "passport-oauth2-refresh": "^1.1.0",
    "passport-openid": "^0.4.0",
    "paypal-rest-sdk": "^1.8.1",
    "popper.js": "^1.15.0",
    "pug": "^2.0.4",
    "request": "^2.88.0",
    "validator": "^11.1.0"
    

