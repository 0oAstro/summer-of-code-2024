# DevClub Summer of Code 2024

Fee fi fo fum, watch out everyone, HERE I COME! DevClub, IIT Delhi welcomes you to the 2nd edition of DevClub Summer of Code. DSoC is a 5-week long hackathon helping students learn development in the fields of **App**, **Backend**, **Frontend**, and **Machine Learning**. These fields are further referred to as tracks.

For each track, there is a **5-week long project** on a common theme divided into weekly guided assignments with increasing difficulty, where we will be providing resources and tasks so that you can learn, build, and showcase!

DevClub will also be giving attracting **rewards** to those people who participate enthusiastically, and will also **recruit members** for the upcoming tenure.

## The Tracks

### [App Development](app)

- [ ] **[Week 1](app/week-1)** :
      Setting up flutter sdk and vscode,
      creating a flutter project,learning basic widgets,routing,snackbars and finally replicating figma design.

- [ ] **[Week 2](app/week-2)** :
      create barcode scanner,learn oops concepts,use provider for state management and create a profile page and add functionalty of logout and profile update.

- [ ] **[Week 3](app/week-3)** :
      Refactoring present codebase to use Firebase services like firebase Authentication,firestore and cloud storage.
      
- [ ] **[Week 4](app/week-4)** :
      Implementing API calls, integrating a payment gateway, generating and managing invoices, updating sales data on Firestore, and integrating Firebase Analytics to track user interactions           and sales       data.Each task focuses on enhancing app functionality and ensuring robust, real-time data handling and analysis.
      
- [ ] **[Week 5](app/week-5)** :
      Implementing password reset and change functionalities using Firebase Auth, enable profile updates, enhance the UI with swipe-to-refresh functionality, and incorporate basic animations in Flutter to improve the overall user experience.        

### [Backend Development](backend)

- [ ] **[Week 1](backend/week-1)** : Setting up the dev environment, initialise a Flask project, connect it to PostgreSQL, and create database models for inventory items and customers.
- [ ] **[Week 2](backend/week-2)** : Create API endpoints, implement CRUD Operations and develop Product Management Interface.
- [ ] **[Week 3](backend/week-3)** : Implement user management and authentication, including secure staff authentication, staff and customer CRUD operations, secure password handling, and transaction management with full CRUD functionality.
- [ ] **[Week 4](backend/week-4)** : Invoice Generation, Azure Account Creation and Setting up Virtual machine.
- [ ] **[Week 5](backend/week-5)** : Setting up the VM, Configuring Ngnix and Gunicorn, Domain Names and HTTPs.

### [Frontend Development](frontend)

- [ ] **[Week 1](frontend/week-1)** : Learn HTML, CSS, and JavaScript to create a visually appealing Login and Sign-Up page. Use provided Figma designs and ensure responsiveness for different screen sizes.
- [ ] **[Week 2](frontend/week-2)** : Use Fetch APIs to authenticate login and store newly signed up members. Get familiar with Bootstrap, and add additional functionalities to the portal using Javascript.
- [ ] **[Week 3](frontend/week-3)** : Learn ReactJS and build the Cashier portal. Use components, React Hooks, React Routers, React-PDF and React-share.
- [ ] **[Week 4](frontend/week-4)** : Get more practice in React JS and build the Admin portal. Use Javascript libraries for analytics and add extra functionalities to improve user experience!
- [ ] **[Week 5](frontend/week-5)** : Deploy your website on the web to complete the final stage of production of the PoS portal. Also try integrating the machine learning model with the frontend to have a complete real-life set-up with price-correction according to the market conditions, etc.

### [Machine Learning](machine-learning)

- [x] **[Week 1](machine-learning/week-1)** : Learn how to build a full ML model to detect fraudulant transactions for our POS system. Start off with standard ML libraries like `sklearn`, `numpy` and `pandas`.
- [ ] **[Week 2](machine-learning/week-2)** : Learn how to build a time-series ML model to forecast sales and inventory for a store! Use time series prediction methods such as ARIMA, LSTM and Prophet.
- [ ] **[Week 3](machine-learning/week-3)** : Learn how to build a clustering model to automatically split your customer base into segments! Then, build a recommmendations engine to give personalised product recommendations.
- [ ] **[Week 4](machine-learning/week-4)** : Learn how to build dynamic pricing models and demand forecasting techniques for our Point of Sale (PoS) application.
- [ ] **[Week 5](machine-learning/week-4)** : Learn how to build an all-powerful AI agentic chatbot that integrates multiple capabilities, including real-time web scraping, sentiment analysis, and a reasoning and planning system using LangChain.

## How to Participate

- First of all, [make a GitHub account](https://github.com/signup) if you haven't already!
- [Fork](https://github.com/devclub-iitd/summer-of-code-2024/fork) this repository. This will give you a copy of this repository where you can do changes as you like. Make sure to [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) latest changes whenever your fork falls behind ours. Later, you should install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), setup [ssh keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent), and [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) your fork on your local machine. You can also refer to [this video](https://youtu.be/YDniPA01pJc?t=2820) by DevClub for learning git.
- Edit [author.json](author.json) to include your details so that we can identify you. Feel free to include other details like `mobile_number`, `email_id`, etc.
- Check out each [track](#the-tracks) from their respective folder, and explore about them. You can try out multiple tracks too!
- Every week, we will share **resources** to learn concepts for each track step-by-step, along with small **assignments** which will add up to the project.
- We will also be having regular online lectures or discussion sessions to help you learn better.
- **Upload** your submissions on your respective forks for each week to the track folder. (NO NEED to create pull request to the main repository for merging your changes).
- Join out **[WhatsApp Community](https://chat.whatsapp.com/EOoXP2jEWAj2V8eJlQqf4H)** for any queries, discussions, and to stay up-to-date.

## Aim

We aim to help students get started with learning coding and software development. The concepts for each [track](#the-tracks) are divided into **weeks** pedagogically so you can learn step-by-step.

No matter if you're a beginner or an expert, at each stage, you will be learning something new, and building something **useful** in real world. You will also be equipped with the skills and tools which would meet **most tech requirements**, and the final projects in each track will be good enough to be put in your **portfolio or CV**.

We are keeping the repository open to **everyone from around the world**, so everyone can use it to learn and share their ideas in the forks.

## Theme and Project

The newly inaugurated retail store at SDA Market, Hauz Khas, is seeking to implement a sophisticated Point of Sales (PoS) system to streamline its inventory management and sales processes. With over 200 Stock Keeping Units (SKUs), the store currently faces challenges in managing inventory and maintaining accurate sales records, leading to significant financial losses. This project aims to develop an integrated PoS system that addresses these challenges by providing a centralized solution for inventory control, customer management, sales tracking, and predictive analytics.

### What is a PoS System

To understand a POS system, it's crucial to grasp the concept of a Point of Sale (POS). A Point of Sale refers to the specific physical or virtual location where a transaction takes place, typically involving the exchange of goods or services for payment. In a physical store, this could be a checkout counter equipped with a cash register and barcode scanner. Online, it represents the virtual platform where customers complete purchases. A POS system encompasses both hardware and software components that streamline these transactions, recording sales data, managing inventory, and often integrating with other business operations like customer relationship management (CRM) and accounting. This technology ensures accuracy, efficiency, and enhanced customer service in retail environments.

### What you will be building

There are basically 4 requirements of the system you will be building:

- Web User Interfaces for Cashier and Admin to record sales and manage inventory and customer data, and a dashboard to view analytics of sales.
- A Mobile App through which the process of billing can be made easily accessible to the cashier by adding advanced features like item barcode scanning, etc.
- A common backend server for the web app and mobile app helping them execute the functionalities. It must be highly secure and role based authentication must be set, i.e. can be accessed by only who is authorised.
- A Machine Learning model which intakes the sales data and provides valuable insights to the store managers regarding sales pattern, inventory holding periods, etc.

## Rewards

- For students from IIT Delhi
  - From each track, we will be recruiting **Executive Members** for the DevClub team in the upcoming tenure.
  - **Certificates** and **ECAs** will be provided to students who perform exceptionally well.
- We will also be sharing **opportunities** relevant to the tools we have covered here, and your submissions can serve as a proof of your skill.
- We will be showcasing the **best projects** on our Social Media handles.

## Connect

Join the DevClub Summer of Code 2024 [WhatsApp Community](https://chat.whatsapp.com/EOoXP2jEWAj2V8eJlQqf4H) to stay updated with releases and involve in project related discussions. Don't forget to introduce yourself - talk about your background in tech, and what are you working on and hoping to learn!

Also, subscribe to DevClub's YouTube channel [@DevClubIITD](https://www.youtube.com/@DevClubIITD) and follow us on Instagram [@devclub_iitd](https://www.instagram.com/devclub_iitd/)

Star ⭐ this repository if you found it useful 😄

## Disclaimer

[DevClub](https://devclub.in/) is an official technical club under [Co-curricular and Academic Interaction Council (CAIC)](https://caic.iitd.ac.in/), [IIT Delhi](https://home.iitd.ac.in). It is also the [Google Developer Student Club (GDSC)](https://gdsc.community.dev/indian-institute-of-technology-delhi/) chapter for Indian Institute of Technology (IIT), Delhi.

We have no other associations with any external organisation. All our learning resources are _free of cost_ for all students.
