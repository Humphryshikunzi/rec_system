I intend to build a personized social recommender, I have these sources:

source 1:
Decoding ML 
Decoding ML 


Discover more from Decoding ML
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Over 25,000 subscribers
Enter your email...
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
Already have an account? Sign in
Building a TikTok-like recommender
Scaling a personalized recommender to millions of items in real-time
Paul Iusztin
Nov 28, 2024

The first lesson of the “Hands-on H&M Real-Time Personalized Recommender” open-source course — a free course that will teach you how to build and deploy a real-time personalized recommender for H&M fashion articles using the 4-stage recommender architecture, the two-tower model design and the Hopsworks AI Lakehouse.

Lessons:
Lesson 1: Building a TikTok-like recommender

Lesson 2: Feature pipelines for TikTok-like recommenders

Lesson 3: Training pipelines for TikTok-like recommenders

Lesson 4: Deploy scalable TikTok-like recommenders

Lesson 5: Using LLMs to build TikTok-like recommenders

🔗 Learn more about the course and its outline.

Lesson 1: Building a TikTok-like recommender
In this lesson, we will discuss the architecture of H&M's real-time personalized recommender. We will use a strategy similar to what TikTok employs for short videos, which will be applied to H&M retail items.

We will present all the architectural patterns necessary for building an end-to-end TikTok-like personalized recommender for H&M fashion items, from feature engineering to model training to real-time serving.

We will teach you how to use the 4-stage architecture to build a system that can handle recommendations from a catalog of millions of items.

We will also walk you through the two-tower model, a flexible neural network design that creates embeddings for users and items.

Ultimately, we will show you how to deploy the entire system using MLOps best practices by leveraging the feature/training/inference (FTI) architecture on top of Hopsworks AI Lakehouse.

By the end of this lesson, you will know what it takes to build a highly scalable and modular real-time personalized recommender on top of H&M data.

In future lessons, we will zoom into the details and code of each H&M real-time personalized recommender component.

💻 Explore all the lessons and the code in our freely available GitHub repository.

Table of Contents
A quick introduction to the H&M retail dataset

Core paradigms for personalized recommendations

Introducing the two-tower embedding model

Understanding the 4-stage recommender architecture

Applying the 4-stage architecture to our H&M use case

Presenting the feature/training/inference (FTI) architecture

Applying the FTI architecture to our retail use case

Deploying the offline ML pipelines using GitHub Actions

Quick demo of the H&M real-time personalized recommender

A quick introduction to the H&M retail dataset
The most standard use case for personalized recommendations is in retail, where you have customers, articles and transactions between the two.

The H&M Personalized Fashion Recommendations dataset [5], which we will use throughout this course, is a perfect example.

It contains the following CSV files:

articles.csv

customers.csv

transactions.csv

We will go deeper into each table in the next lesson when we will design the features.

When it comes to gathering custom data for personalized recommendations, the most challenging part is to get (or generate) meaningful interactions between a customer and an item, such as when a customer:

clicked on an item;

added an item to the cart;

bought an item.

Thus, we will leverage the transactions provided by the H&M dataset to train our models and present our use case.

But, to mimic a real-world scenario, we will gather new interactions from our PoC UI, which will influence the following predicted recommendations.

Core paradigms for personalized recommendations
When it comes to recommendations, you can choose between two core paradigms:

Content-based filtering: This approach recommends items by analyzing the features or characteristics of items a user has previously interacted with, then finding new items with similar features – for example, if a customer frequently buys floral dresses, the system would recommend other floral-patterned clothing items.

Collaborative filtering: This approach makes recommendations by identifying patterns in user-item interactions and finding similar users or items based on their behavior patterns. For instance, if customers who buy leather jackets also tend to buy black boots, the system would recommend black boots to new customers who purchase leather jackets.


Figure 1: Core paradigms
Let’s see how we can apply these two paradigms using the two-tower model.

Introducing the two-tower embedding model
The first step in understanding how a neural network-based recommender works is to examine the architecture of the two-tower embedding model.

At its core, the two-tower model architecture aims to compute feature-rich embeddings for the customers and items in the same embedding space. Thus, when looking for recommendations for a customer, we can calculate the distance of the customer’s embeddings and the items to search for the most relevant item candidates [8].


Figure 2: The two-tower model
The two-tower model architecture trains two neural networks in parallel:

The customer query encoder transforms customer features into a dense embedding vector.

The item candidates encoder transforms item features into dense embeddings in the same vector space as the customer embeddings.

Both encoders can process various types of features:

Customer encoder: demographic information, historical behavior, contextual features

Item encoder: tags, description, rating

This introduces a content-based paradigm. Similar items and customers will be clustered together if enough features are used.

A key distinction from traditional architectures is that the two-tower model processes user and item features separately. This makes it highly efficient for large-scale retrieval since item embeddings can be pre-computed and stored in an approximate nearest neighbor (ANN) index or database (also known as vector databases).

Using the dot product as a score for the loss function, where we expect a 1 when a customer interacts with an item and a 0 when there is no interaction, we indirectly use the cosine distance, which forces the two embeddings to be in the same vector space.

cos distance = dot product with normalized vectors

Using a dot product as a score for the loss function introduces a collaborative filtering paradigm because it captures customer-item interaction patterns. Customers with similar behaviors and items accessed in the same pattern will be clustered.

Thus, depending on how many features you use for the items and customers, the two-tower model can be only a collaborative filtering algorithm (if only the IDs are used) or both if there is enough signal in the provided features.

We will dig into the architecture of the two encoders and how they are trained in Lesson 3, explaining the training pipeline.

Let’s intuitively understand how these two models are used in the 4-stage recommender architecture.

Understanding the 4-stage recommender architecture
The 4-stage recommender architecture is the standard for building scalable, real-time personalized recommenders based on various data types and use cases.

It’s used and proposed by giants such as Nvidia [7] and YouTube [2].

In the 4-stage recsys architecture, the data flows in two ways:

An offline pipeline that computes the candidate embeddings and loads them to a vector index or database. This pipeline usually runs in batch mode.

An online pipeline that computes the actual recommendations for a customer. This pipeline can run in batch, async, real-time or streaming mode, depending on the type of application you build.

Computing the item candidate embeddings offline allows us to make recommendations from a large corpus (millions) of items while still being confident that the small number of recommended items is personalized and engaging for the user.


Figure 3: Data flow of the 4-stage recommender.
The offline pipeline leverages the Items Candidate Encoder Model (trained using the Two Tower model) to compute embeddings for all the items in our database. It loads the item embeddings and their metadata, such as the ID, into an approximate nearest neighbor (ANN) index optimized for low-latency retrieval. The ANN indexes come in two flavors:

vector index (e.g., ScaNN, Faiss);

vector database (e.g., Hopsworks, Qdrant, MongoDB).

By decoupling the item embedding creation from the actual recommendation, we can drastically speed up the recommendation for each customer as:

Everything we want to find (recommend) is precomputed when customers access our application.

We can optimize the offline and online pipelines differently for better latency, lower costs, required throughput, etc.

The online pipeline is split into 4-stages (as the name suggests), starting with the user’s requests and ending with the recommendations.


Figure 4: The 4-stage recommender architecture
Stage 1
This stage aims to process a large (>100M elements up to millions) corpus of candidate items and retrieve a relevant subset (~hundreds) of items for downstream ranking and filtering tasks.

The candidate generation step only provides broad personalization via collaborative filtering. Similarities are expressed in coarse features such as item and customer IDs.

The pipeline takes a customer_id and other input features, such as the current date, computes the customer embedding using the Customer Query Model (trained using the Two Tower model), and queries the vector DB for similar candidate items.

Using the customer’s embedding, the vector DB (or index) scans the entire corpus and reduces it to xN potential candidates (~hundreds).

Stage 2
Stage 2 takes the N candidate items and applies various filters, such as removing items already seen or purchased.

The core idea is to filter out unnecessary candidates before proceeding to the most expensive operations from Stage 3. The filtering is often done using a Bloom filter, a space-efficient probabilistic data structure used to test whether an element is a set member (such as seen or purchased items).

After this stage, we are left with only xM item candidates.

Stage 3
Stage 3 takes the xM item candidates and prepares them for ranking. An algorithm that provides a score for each “(item candidate, customer)” tuple based on how relevant that item is to a particular customer.

During ranking, we can access more features describing the item and the user’s relationship, as only a few hundred items are being scored rather than the millions scored in candidate generation.

The ranking step is slower as we enhance the items and customers with multiple features. We usually use a feature store to query all the necessary features.

Thus, extra I/O overhead is added by querying the feature store, and the ranking algorithm is slower as it works with more data.

The ranking model can use a boosting tree, such as XGBoost or CatBoost, a neural network or even an LLM.

Presenting a few “best” recommendations in a list requires a fine-level representation to distinguish relative importance among the candidate items. The ranking network accomplishes this task by assigning a score to each item using a rich set of features describing the item and user.

Stage 4
After the ranking model scores each “(item candidate, customer)” tuple, we must order the items based on the ranking score plus other optional business logic.

The highest-scoring items are presented to the user and ranked by their score.

If the items candidate list is too extensive for our use case, we could further cut it to xK item candidates.

It is critical to order the items based on relevance. Having the most personalized candidates at the top increases the customer's probability of clicking on them.

For example, you want your No. 1 movie or playlist always to be the first thing when you open Netflix, YouTube or Spotify. You don’t want to explore too much until you find it.

By the end of Stage 4, we will have xK relevant and personalized items that we can display in our application as needed.

Let’s apply it to our H&M use case to understand how this works fully.

Applying the 4-stage architecture to our H&M use case
If we understand how the two-tower model and 4-stage architecture work, applying it to our H&M use case is very intuitive.

First, let’s understand who the “customers” and “items” are in our use case.

The customers are the users looking to buy items on the H&M site or application.

The items are the fashion items sold by H&M, such as clothes, socks, shoes, etc.

Thus, we must show the customers fashion items they are most likely to buy.

For example, if he searched for T-shirts, most likely we should recommend T-shirts. Our recsys should pick up on that.


Figure 5: The 4-stage recommender architecture applied to our H&M use case
Secondly, let’s look at a concrete flow of recommending H&M articles:

While a customer surfs the H&M app, we send its ID and date to the recsys inference pipeline.

The customer query model computes the customer’s embedding based on the two features from 1.

As the customer’s embedding is in the same vector space as the H&M fashion items, we leverage a Hopsworks vector index to retrieve a coarse list of relevant articles.

Next, we filtered out all the items the customer already clicked on or bought.

We enhance the fashion articles and customer with a more extensive list of features from our Hopsworks feature views.

We use a CatBoost model to rank the remaining fashion items relative to the customer.

We sort the articles based on the relevance score and show them to the customer.

But what is Hopsworks?
It’s an AI Lakehouse that will help us ship the recsys to production.

It provides the following capabilities:

Feature store: Store, version, and access the features required for training (offline, high throughput) and inference (online, low latencies). More on feature stores [11].

Model registry: Store, version, and access the models (candidate encoder, query encoder, ranking model).

Serving layer: Host the inference pipeline containing the 4 steps to make real-time predictions.

Given this, we can store our features in Hopsworks, make them available for training and inference, and deploy our models to production by leveraging their model registry and serving layer.

Click here to find out more about Hopsworks - The AI Lakehouse.

Let’s quickly present the FTI architecture and, in more detail, how we used Hopsworks to ship our recsys app.

Presenting the feature/training/inference (FTI) architecture
The pattern suggests that any ML system can be boiled down to these three pipelines: feature, training, and inference.

Jim Dowling, CEO and Co-Founder of Hopsworks introduced the pattern to simplify building production ML systems [3, 4].

The feature pipelines take raw data as input and output features and labels to train our model(s).

The training pipeline takes the features and labels from the feature stored as input and outputs our trained model(s).

The inference pipeline inputs the features & labels from the feature store and the trained model(s) from the model registry. With these two, predictions can be easily made in either batch or real-time mode.


Figure 6: The feature/training/inference (FTI) architecture
To conclude, the most important thing you must remember about the FTI pipelines is their interface:

The feature pipeline takes in data and outputs features & labels saved to the feature store.

The training pipelines query the features store for features & labels and output a model to the model registry.

The inference pipeline uses the features from the feature store and the model from the model registry to make predictions.

It doesn’t matter how complex your ML system gets. These interfaces will remain the same.

There is a lot more to the FTI architecture. Consider reading this article [6] for a quick introduction or a more in-depth series on scaling ML pipelines using MLOps best practices, starting here [12].

Applying the FTI architecture to our retail use case
The final step in understanding the architecture of the H&M recsys is presenting how we can apply the FTI pattern to it.

This pattern will help us move from Notebooks to production by deploying our offline ML pipelines and serving the inference pipeline in real time (with the 4-stage logic).

The ML pipelines (feature, training, embeddings, inference) will be implemented in Python. Meanwhile, we will leverage the Hopsworks AI Lakehouse for storage and deployment.

Let’s see how we can do that by zooming in each pipeline independently.

The feature pipeline transforms raw H&M data (usually stored in a data warehouse) into features stored in Hopsworks feature groups.

We will detail the features and what a feature group is in Lesson 2. For now, you have to know that a feature group is similar to a table in a database, where we group related features (e.g., customers, articles, transactions, etc.). More on feature groups [9].


Figure 7: The architecture of the H&M real-time personalized recommender - Powered by Hopsworks
The training pipeline inputs the features from various Hopsworks feature views, trains the two-tower and ranking models, and saves them in the Hopsworks model registry.

Remember that the two-tower model trains two models in parallel: the items candidate and query encoders. Thus, we save them independently in the model registry, as we will use them at different times.

A feature view is a virtual table for read-only operations (training, inference). It is created based on multiple features picked from multiple feature groups. Doing so allows you to create virtual tables with the exact features you need for training (offline mode) or inference (online mode). More on feature views [10].

The embeddings inference pipeline (offline) loads the candidate model from the model registry and fashion items from the retrieval feature view, computes the embeddings, and loads them to the candidate embeddings Hopsworks vector index (also a feature group).

Notice how the embedding pipeline follows the interface of the inference pipeline proposed by the FTI architecture.

This is because the inference logic is split into offline and online pipelines, as discussed in the 4-stage recsys architecture section.

This highlights that the FTI pipelines are not only three pipelines but a mindmap for modeling your system, which usually contains many more components.

Ultimately, the real-time inference pipeline (online) loads the query retrieval and ranking models from the model registry and their associated features from the Hopsworks feature view.

This pipeline is deployed on Hopsworks AI Lakehouse as a real-time API called from the front end through HTTP requests.

The real-time inference pipeline wraps the 4-stage recsys logic, which serves as the final personalized recommendation for the customer.

We will provide more details about the serving infrastructure in Lesson 4.

The feature, training, and embedding inference pipelines run offline. Thus, we can leverage other tools to run them based on different triggers to update the features, models, and item candidates.

One option is GitHub Actions.

Deploying the offline ML pipelines using GitHub Actions
Following the FTI architecture, the ML pipelines are completely decoupled and can be run as independent components if we respect a particular order.

Thus, together with Hopsworks as an AI lakehouse, we can quickly ship the ML pipelines to GitHub Actions, which can run on a:

manual trigger;

schedule;

after merging a new feature branch in the main branch (or staging).


Figure 8: Deploying the offline ML pipelines using GitHub Actions
Because our models are small, we can use GitHub Actions for free computing. Thus, training them on a CPU is feasible.

Also, as GitHub Actions is well integrated with your code, with just a few lines of code, we can prepare the necessary Python environment, run the code, and chain the ML pipelines as a direct acyclic graph (DAG).

We will detail the implementation in Lesson 4.

Quick demo of the H&M real-time personalized recommender
To show an end-to-end PoC of our H&M real-time personalized recommender that is ready for production, we have used the following tech stack:

Hopsworks (serverless platform) offers a freemium plan to host our feature store, model registry, and real-time serving layer.

GitHub Actions to host and schedule our offline ML pipelines (as explained in the section above)

Streamlit to prototype a simple frontend to play around with the recommender. Also, we leverage Stream Cloud to host the frontend.

Will this cost me money? We will stick to the free tier for all these tools and platforms, allowing us to test the whole recsys series end-to-end at no cost.


Figure 9: Streamlit application powered by real-time personalized recommendations.
To quickly test things out, follow the documentation from GitHub on how to set up Hopsworks, GitHub Actions, and Streamlit and run the entire recsys application.

Conclusion
This lesson taught us about the two-tower model, 4-stage recsys architecture, and the FTI pattern.

Then, we saw how to apply these patterns to our H&M use case.

In Lesson 2, we will start zooming in on the feature pipeline and Hopsworks, detailing the features we use for the two-tower and ranking models and the code.

💻 Explore all the lessons and the code in our freely available GitHub repository.

If you have questions or need clarification, feel free to ask. See you in the next session!

References
Literature
[1] Decodingml. (n.d.). GitHub - decodingml/personalized-recommender-course. GitHub. https://github.com/decodingml/personalized-recommender-course

[2] Covington, P., Adams, J., & Sargin, E. (n.d.). Deep Neural Networks for YouTube Recommendations. Google Research. https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

[3] Dowling, J. (2024a, August 5). Modularity and Composability for AI Systems with AI Pipelines and Shared Storage. Hopsworks. https://www.hopsworks.ai/post/modularity-and-composability-for-ai-systems-with-ai-pipelines-and-shared-storage

[4] Dowling, J. (2024b, November 1). From MLOps to ML Systems with Feature/Training/Inference Pipelines. Hopsworks. https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines

[5] H&M personalized fashion recommendations. (n.d.). Kaggle. https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

[6] Iusztin, P. (2024, August 10). Building ML system using the FTI architecture. Decoding ML Newsletter. https://decodingml.substack.com/p/building-ml-systems-the-right-way

[7] NVIDIA Merlin Recommender System Framework. (n.d.). NVIDIA Developer. https://developer.nvidia.com/merlin

[8] Wortz, J., & Totten, J. (2023, April 19). Tensorflow deep retrieval using Two Towers architecture. Google Cloud Blog. https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture

[9] Hopsworks. (n.d.). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/fs/feature_group/fg_overview/

[10] Hopsworks. (n.d.-b). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/fs/feature_view/fv_overview/

[11] Hopsworks. (n.d.-a). What is a Feature Store: The Definitive Guide - Hopsworks. https://www.hopsworks.ai/dictionary/feature-store

[12] Hopsworks. (n.d.-b). What is a Machine Learning Pipeline? - Hopsworks. https://www.hopsworks.ai/dictionary/ml-pipeline

Images
If not otherwise stated, all images are created by the author.

Sponsors
Thank our sponsors for supporting our work!


Subscribe to Decoding ML
Launched 2 years ago
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Type your email...
Subscribe
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
106 Likes
∙
15 Restacks
Discussion about this post
Write a comment...
Miguel Otero Pedrido
Nov 28

Amazing introduction to the 4 stage design Paul! 💪

Like (3)
Reply
Share
1 reply by Paul Iusztin
Jyothi Vishnu Vardhan kolla
Dec 1

This was such a great resource, Thanks for all the hard work.

Like (2)
Reply
Share
2 replies by Paul Iusztin and others
8 more comments...

Build your Second Brain AI assistant
Using agents, RAG, LLMOps and LLM systems
Feb 6 • Paul Iusztin
870
35

LLMOps for production agentic RAG
Evaluating and monitoring LLM agents with SmolAgents and Opik
Mar 20 • Paul Iusztin and Anca Ioana Muscalagiu
94

Playbook to fine-tune and deploy LLMs
Specialized open-source LLMs for production
Mar 6 • Paul Iusztin
90
4

Ready for more?

Type your email...
Subscribe
© 2025 Paul Iusztin
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture

source 2:
Decoding ML 
Decoding ML 

Feature pipelines for TikTok-like recommenders
Feature engineering for a H&M real-time personalized recommender
Paolo Perrone
Dec 05, 2024

The second lesson of the “Hands-On Real-time Personalized Recommender” open-source course — a free course that will teach you how to build and deploy a production-ready real-time personalized recommender for H&M articles using the four-stage recsys architecture, the two-tower model design and the Hopsworks AI Lakehouse.

Lessons:
Lesson 1: Building a TikTok-like recommender

Lesson 2: Feature pipelines for TikTok-like recommenders

Lesson 3: Training pipelines for TikTok-like recommenders

Lesson 4: Deploy scalable TikTok-like recommenders

Lesson 5: Using LLMs to build TikTok-like recommenders

🔗 Learn more about the course and its outline.


Figure 1: The Feature Pipeline in the FTI Architecture
Lesson 2: Feature pipelines for TikTok-like recommenders
In this lesson, we’ll explore the feature pipeline that forms the backbone of a real-time personalized recommender using the H&M retail dataset.

Our primary focus will be on the steps involved in creating and managing features, which are essential for training effective machine learning models.

By the end of this lesson, you will have a thorough understanding of how to:

present and process the H&M dataset,

engineer features for both the retrieval and ranking models,

create and manage Hopsworks Feature Groups for efficient ML workflow,

lay the groundwork for future steps, such as integrating streaming pipelines to enable real-time data processing and recommendations.

💻 Explore all the lessons and the code in our freely available GitHub repository.


Figure 2: The Batch Feature Pipeline in the H&M recommender
We need to set up the environment to start building the feature pipeline.

The following code initializes the notebook environment, checks if it’s running locally or in Google Colab, and configures the Python path to import custom modules.

import sys
from pathlib import Path

def is_google_colab() -> bool:
    if "google.colab" in str(get_ipython()):
        return True
    return False

def clone_repository() -> None:
    !git clone https://github.com/decodingml/hands-on-recommender-system.git
    %cd hands-on-recommender-system/

def install_dependencies() -> None:
    !pip install --upgrade uv
    !uv pip install --all-extras --system --requirement pyproject.toml

if is_google_colab():
    clone_repository()
    install_dependencies()
    root_dir = str(Path().absolute())
    print("⛳️ Google Colab environment")
else:
    root_dir = str(Path().absolute().parent)
    print("⛳️ Local environment")

# Add the root directory to the PYTHONPATH
if root_dir not in sys.path:
    print(f"Adding the following directory to the PYTHONPATH: {root_dir}")
    sys.path.append(root_dir)
🔗 Full code here → Github

Table of Contents
1 - The H&M dataset

2 - Feature engineering

3 - Creating Feature Groups in Hopsworks

4 - Next steps: Implementing a streaming data pipeline

5 - Running the feature pipeline

1 - The H&M dataset
Before diving into feature engineering, let’s first take a closer look at the H&M Fashion Recommendation dataset.

The dataset consists of three main tables: articles, customers, and transactions.

Below is how you can extract and inspect the data:

from recsys.raw_data_sources import h_and_m as h_and_m_raw_data

# Extract articles data
articles_df = h_and_m_raw_data.extract_articles_df()
print(articles_df.shape)
articles_df.head()

# Extract customers data
customers_df = h_and_m_raw_data.extract_customers_df()
print(customers_df.shape)
customers_df.head()

# Extract transactions data
transactions_df = h_and_m_raw_data.extract_transactions_df()
print(transactions_df.shape)
transactions_df.head()
🔗 Full code here → Github

This is what the data looks like:

1 - Customers Table

Customer ID: A unique identifier for each customer.

Age: Provides demographic information, which can help predict age-related purchasing behavior.

Membership status: Indicates whether a customer is a member, which may impact buying patterns and preferences.

Fashion news frequency: Reflect how often customers receive fashion news, hinting at their engagement level.

Club member status: Show if the customer is an active club member, which can affect loyalty and purchase frequency.

FN (fashion news score): A numeric score reflecting customer's engagement with fashion-related content.

2 - Articles Table

Article ID: A unique identifier for each product.

Product group: Categorizes products into groups like dresses, tops, or shoes.

Color: Describes each product's color, which is important for visual similarity recommendations.

Department: Indicates the department to which the article belongs, providing context for the type of products.

Product type: A more detailed classification within product groups.

Product code: A unique identifier for each product variant.

Index code: Represents product indexes, useful for segmenting similar items within the same category.

3 - Transactions Table

Transaction ID: A unique identifier for each transaction.

Customer ID: Links the transaction to a specific customer.

Article ID: Links the transaction to a specific product.

Price: Reflect the transaction amount, which helps analyze spending habits.

Sales channel: Shows whether the purchase was made online or in-store.

Timestamp: Records the exact time of the transaction, useful for time-based analysis.

🔗 Full code here → Github

Tables Relationships

The tables are connected through unique identifiers like customer and article IDs. These connections are crucial for making the most of the H&M dataset:

Customer to Transactions: By associating customer IDs with transaction data, we can create behavioral features like purchase frequency, recency, and total spending, which provide insights into customer activity and preferences.

Articles to Transactions: Linking article IDs to transaction records helps us analyze product popularity, identify trends, and understand customer preferences for different types of products.

Cross-Table Analysis: Combining data from multiple tables allows us to perform advanced feature engineering. For example, we can track seasonal product trends or segment customers based on purchasing behavior, enabling more personalized recommendations.

Table relationships provide a clearer picture of how customers interact with products, which helps improve the accuracy of the recommendation model in suggesting relevant items.


Figure 3: The H&M Personalized Fashion Recommendations Dataset
The Customers table contains customer data, including unique customer IDs (Primary Key), membership status, and fashion news preferences.

The Articles table stores product details like article IDs (Primary Key), product codes, and product names.

The Transactions table links customers and articles through purchases, with fields for the transaction date, customer ID (Foreign Key), and article ID (Foreign Key).

The double-line notations between tables indicate one-to-many relationships: each customer can make multiple transactions, and each transaction can involve multiple articles.

2 - Feature engineering
The feature pipeline takes as input raw data and outputs features and labels used for training and inference.

📚 Read more about feature pipelines and their integration into ML systems [6].

Creating effective features for both retrieval and ranking models is the foundation of a successful recommendation system.

Feature engineering for the two-tower model
The two-tower retrieval model's primary objective is to learn user and item embeddings that capture interaction patterns between customers and articles.

We use the transactions table as our source of ground truth - each purchase represents a positive interaction between a customer and an article.

This is the foundation for training the model to maximize similarity between embeddings for actual interactions (positive pairs).

The notebook imports necessary libraries and modules for feature computation.

This snippet lists the default settings used throughout the notebook, such as model IDs, learning rates, and batch sizes.

It is helpful for understanding the configuration of the feature pipeline and models.

pprint(dict(settings))

🔗 Full code here → Github

Training objective

The goal of the two-tower retrieval model is to use a minimal, strong feature set that is highly predictive but does not introduce unnecessary complexity.

The model aims to maximize the similarity between customer and article embeddings for purchased items while minimizing similarity for non-purchased items.

This objective is achieved using a loss function such as cross-entropy loss for sampled softmax, or contrastive loss. The embeddings are then optimized for nearest-neighbor search, which enables efficient filtering in downstream recommendation tasks.


Figure 4: The retrieval dataset for training the two-tower network
Feature selection

The two-tower retrieval model intentionally uses a minimal set of strong features to learn robust embeddings:

Query features - used by the QueryTower (the customer encoder from the two-tower model):

customer_id: A categorical feature that uniquely identifies each user. This is the backbone of user embeddings.

age: A numerical feature that can capture demographical patterns.

month_sin and month_cos: Numerical features that encode cyclic patterns (e.g., seasonality) in user behavior.

Candidate features - used by the ItemTower (the H&M fashion articles encoder from the two-tower model):

article_id: A categorical feature that uniquely identifies each item. This is the backbone of item embeddings.

garment_group_name: A categorical feature that captures high-level categories (e.g., "T-Shirts", "Dresses") to provide additional context about the item.

index_group_name: A categorical feature that captures broader item groupings (e.g., "Menswear", "Womenswear") to provide further context.

These features are passed through their respective towers to generate the query (user) and item embeddings, which are then used to compute similarities during retrieval.

The limited feature set is optimized for the retrieval stage, focusing on quickly identifying candidate items through an approximate nearest neighbor (ANN) search.

This aligns with the four-stage recommender system architecture, ensuring efficient and scalable item retrieval.

This snippet computes features for articles, such as product descriptions and metadata, and displays their structure.

articles_df = compute_features_articles(articles_df)
articles_df.shape
articles_df.head(3)

compute_features_articles() takes the articles dataframe and transforms it into a dataset with 27 features across 105,542 articles.

import polars as pl

def compute_features_articles(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            get_article_id(df).alias("article_id"),
            create_prod_name_length(df).alias("prod_name_length"),
            pl.struct(df.columns)
            .map_elements(create_article_description)
            .alias("article_description"),
        ]
    )

    # Add full image URLs.
    df = df.with_columns(image_url=pl.col("article_id").map_elements(get_image_url))

    # Drop columns with null values
    df = df.select([col for col in df.columns if not df[col].is_null().any()])

    # Remove 'detail_desc' column
    columns_to_drop = ["detail_desc", "detail_desc_length"]
    existing_columns = df.columns
    columns_to_keep = [col for col in existing_columns if col not in columns_to_drop]

    return df.select(columns_to_keep)
One standard approach when manipulating text before feeding it into a model is to embed it. This solves the curse of dimensionality or information loss from solutions such as one-hot encoding or hashing.

The following snippet generates embeddings for article descriptions using a pre-trained SentenceTransformer model.

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Loading '{settings.FEATURES_EMBEDDING_MODEL_ID}' embedding model to {device=}")

# Load the embedding model
model = SentenceTransformer(settings.FEATURES_EMBEDDING_MODEL_ID, device=device)

# Generate embeddings for articles
articles_df = generate_embeddings_for_dataframe(
    articles_df, "article_description", model, batch_size=128
)
🔗 Full code here → Github

Features engineering for the Ranking model
The ranking model has a more complex objective: accurately predicting the likelihood of purchase for each retrieved item.

This model uses a combination of query and item features, along with labels, to predict the likelihood of interaction between users and items.

This feature set is designed to provide rich contextual and descriptive information, enabling the model to rank items effectively.

Generate features for customers:

Training objective

The model is trained to predict purchase probability, with actual purchases (from the Transactions table) serving as positive labels (1) and non-purchases as negative labels (0).

This binary classification objective helps order retrieved items by their likelihood of purchase.


Figure 5: Ranking dataset for training the ranking model.
Feature selection

Query Features - identical to those used in the Retrieval Model to encode the customer

Item Features - used to represent the articles in the dataset. These features describe the products' attributes and help the model understand item properties and relationships:

article_id: A categorical feature that uniquely identifies each item, forming the foundation of item representation.

product_type_name: A categorical feature that describes the specific type of product (e.g., "T-Shirts", "Dresses"), providing detailed item-level granularity.

product_group_name: A categorical feature for higher-level grouping of items, useful for capturing broader category trends.

graphical_appearance_name: A categorical feature representing the visual style of the item (e.g., "Solid", "Striped").

colour_group_name: A categorical feature that captures the color group of the item (e.g., "Black", "Blue").

perceived_colour_value_name: A categorical feature describing the brightness or value of the item's color (e.g., "Light", "Dark").

perceived_colour_master_name: A categorical feature representing the master color of the item (e.g., "Red", "Green"), providing additional color-related information.

department_name: A categorical feature denoting the department to which the item belongs (e.g., "Menswear", "Womenswear").

index_name: A categorical feature representing broader categories, providing a high-level grouping of items.

index_group_name: A categorical feature that groups items into overarching divisions (e.g., "Divided", "H&M Ladies").

section_name: A categorical feature describing the specific section within the store or catalog.

garment_group_name: A categorical feature that captures high-level garment categories (e.g., "Jackets", "Trousers"), helping the model generalize across similar items.

Label - A binary feature used for supervised learning

`1` indicates a positive pair (customer purchased the item).

`0` indicates a negative pair (customer did not purchase the item, randomly sampled).

This approach is designed for the ranking stage of the recommender system, where the focus shifts from generating candidates to fine-tuning recommendations with higher precision.

By incorporating both query and item features, the model ensures that recommendations are relevant and personalized.

Constructing the final ranking dataset
The ranking dataset is the final dataset used to train the scoring/ranking model in the recommendation pipeline.

It is computed by combining query (customer) features, item (article) features, and the interactions (transactions) between them.

The compute_ranking_dataset() combines the different features from the Feature Groups:

`trans_fg`: The transactions Feature Group, which provides the labels (`1` for positive pairs and `0` for negative pairs) and additional interaction-based features (e.g., recency, frequency).

`articles_fg`: The articles Feature Group, which contains the engineered item features (e.g., product type, color, department, etc.).

`customers_fg`: The customers Feature Group, which contains customer features (e.g., age, membership status, purchase behavior).

The resulting ranking dataset includes:

Customer Features: From `customers_fg`, representing the query.

Item Features: From `articles_fg`, representing the candidate items.

Interaction Features: From `trans_fg`, such as purchase frequency or recency, which capture behavioral signals.

Label: A binary label (`1` for purchased items, `0` for negative samples).

The result is a dataset where each row represents a customer-item pair, with the features and label indicating whether the customer interacted with the item.

In practice, this looks like this:

ranking_df = compute_ranking_dataset(
    trans_fg,
    articles_fg,
    customers_fg,
)
ranking_df.shape
Negative sampling for the ranking dataset

The ranking dataset includes both positive and negative samples.

This ensures the model learns to differentiate between relevant and irrelevant items:

Positive samples (Label = 1): derived from the transaction Feature Group (`trans_fg`), where a customer purchased a specific item.

Negative samples (Labels = 0): generated by randomly sampling items the customer did not purchase. These represent items the customer is less likely to interact with and help the model better understand what is irrelevant to the user.

# Inspect the label distribution in the ranking dataset
ranking_df.get_column("label").value_counts()
Outputs:

label	count
i32	u32
1	20377
0	203770
Negative Samples are constrained to make them realistic, such as sampling items from the same category or department as the customer's purchases or including popular items the customer hasn’t interacted with, simulating plausible alternatives.

For example, if the customer purchased a "T-shirt," negative samples could include other "T-shirts” they didn't buy.

Negative samples are often balanced in proportion to positive ones. For every positive sample, we might add 1 to 5 negative ones. This prevents the model from favoring negative pairs, which are much more common in real-world data.

import polars as pl

def compute_ranking_dataset(trans_fg, articles_fg, customers_fg) -> pl.DataFrame:
    ... # More code

    # Create positive pairs
    positive_pairs = df.clone()

    # Calculate number of negative pairs
    n_neg = len(positive_pairs) * 10

     # Create negative pairs DataFrame
    article_ids = (df.select("article_id")
                    .unique()
                    .sample(n=n_neg, with_replacement=True, seed=2)
                    .get_column("article_id"))
    
    customer_ids = (df.select("customer_id")
                     .sample(n=n_neg, with_replacement=True, seed=3)
                     .get_column("customer_id"))

    other_features = (df.select(["age"])
                       .sample(n=n_neg, with_replacement=True, seed=4))

    # Construct negative pairs
    negative_pairs = pl.DataFrame({
        "article_id": article_ids,
        "customer_id": customer_ids,
        "age": other_features.get_column("age"),
    })

    # Add labels
    positive_pairs = positive_pairs.with_columns(pl.lit(1).alias("label"))
    negative_pairs = negative_pairs.with_columns(pl.lit(0).alias("label"))

    # Concatenate positive and negative pairs
    ranking_df = pl.concat([
        positive_pairs,
        negative_pairs.select(positive_pairs.columns)
    ])

    ... More code

    return ranking_df
3 - Creating Feature Groups in Hopsworks
Once the ranking dataset is computed, it is uploaded to Hopsworks as a new Feature Group, with lineage information reflecting its dependencies on the parent Feature Groups (`articles_fg`, `customers_fg`, and `trans_fg`).

logger.info("Uploading 'ranking' Feature Group to Hopsworks.")
rank_fg = feature_store.create_ranking_feature_group(
    fs,
    df=ranking_df,
    parents=[articles_fg, customers_fg, trans_fg],
    online_enabled=False
)
logger.info("✅ Uploaded 'ranking' Feature Group to Hopsworks!!")
This lineage ensures that any updates to the parent Feature Groups (e.g., new transactions or articles) can be propagated to the ranking dataset, keeping it up-to-date and consistent.

The Hopsworks Feature Store is a centralized repository for managing features.

The following shows how to authenticate and connect to the feature store:

from recsys import hopsworks_integration

# Connect to Hopsworks Feature Store
project, fs = hopsworks_integration.get_feature_store()
🔗 Full code here → Github

Step 1: Define Feature Groups

Feature Groups are logical groupings of related features that can be used together in model training and inference.

For example:

1 - Customer Feature Group

Includes all customer-related features, such as demographic, behavioral, and engagement metrics.

Demographics: Age, gender, membership status.

Behavioral features: Purchase history, average spending, visit frequency.

Engagement metrics: Fashion news frequency, club membership status.

2 - Article Feature Group

It includes features related to articles (products), such as descriptive attributes, popularity metrics, and image features.

Descriptive attributes: Product group, color, department, product type, product code.

Popularity metrics: Number of purchases, ratings.

Image features: Visual embeddings derived from product images.

3 - Transaction Feature Group

Includes all transaction-related features, such as transactional details, interaction metrics, and contextual features.

Transactional attributes: Transaction ID, customer ID, article ID, price.

Interaction metrics: Recency and frequency of purchases.

Contextual features: Sales channel, timestamp of transaction.

Adding a feature group to Hopsworks:

from recsys.hopsworks_integration.feature_store import create_feature_group

# Create a feature group for article features
create_feature_group(
    feature_store=fs,
    feature_data=article_features_df,
    feature_group_name="articles_features",
    description="Features for articles in the H&M dataset"
)
🔗 Full code here → Github

Step 2: Data ingestion
To ensure the data is appropriately structured and ready for model training and inference, the next step involves loading data from the H&M dataset into the respective Feature Groups in Hopsworks.

Here’s how it works:

1 - Data loading

Start by extracting data from the H&M source files, processing them into features and loading them into the correct Feature Groups.

2 - Data validation

After loading, check that the data is accurate and matches the expected structure.

Consistency checks: Verify the relationships between datasets are correct.

Data cleaning: Address any issues in the data, such as missing values, duplicates, or inconsistencies.

Luckily, Hopsworks supports integration with Great Expectations, adding a robust data validation layer during data loading.

Step 3: Versioning and metadata management
Versioning and metadata management are essential for keeping your Feature Groups organized and ensuring models can be reproduced.

The key steps are:

Version control: Track different versions of Feature Groups to ensure you can recreate and validate models using specific data versions. For example, if there are significant changes to the Customer Feature Group, create a new version to reflect those changes.

Metadata management: Document the details of each feature, including its definition, how it’s transformed, and any dependencies it has on other features.

rank_fg = fs.get_or_create_feature_group(
        name="ranking",
        version=1,
        description="Derived feature group for ranking",
        primary_key=["customer_id", "article_id"],
        parents=[articles_fg, customers_fg, trans_fg],
        online_enabled=online_enabled,
    )
rank_fg.insert(df, write_options={"wait_for_job": True})

for desc in constants.ranking_feature_descriptions:
        rank_fg.update_feature_description(desc["name"], desc["description"])
Defining Feature Groups, managing data ingestion, and tracking versions and metadata ensure your features are organized, reusable, and reliable, making it easier to maintain and scale your ML workflows.

View results in Hopsworks Serverless: Feature Store → Feature Groups

The importance of Hopsworks Feature Groups
Hopsworks Feature Groups are key in making machine learning workflows more efficient and organized.

Here’s how they help:

1 - Centralized repository

Single source of truth: Feature Groups in Hopsworks provide a centralized place for all your feature data, ensuring everyone on your team uses the same, up-to-date data. This reduces the risk of inconsistencies and errors when different people use outdated or other datasets.

Easier management: Managing all features in one place becomes easier. Updating, querying, and maintaining the features is streamlined, leading to increased productivity and smoother workflows.

2- Feature reusability

Cross-model consistency: Features stored in Hopsworks can be used across different models and projects, ensuring consistency in their definition and application. This eliminates the need to re-engineer features each time, saving time and effort.

Faster development: Since you can reuse features, you don’t have to start from scratch. You can quickly leverage existing, well-defined features, speeding up the development and deployment of new models.

3 - Scalability

Optimized Performance: The platform ensures that queries and feature updates are performed quickly, even when dealing with large amounts of data. This is crucial for maintaining model performance in production.

4 - Versioning and lineage

Version control: Hopsworks provides version control for Feature Groups, so you can keep track of changes made to features over time. This helps reproducibility, as you can return to previous versions if needed.

Data lineage: Tracking data lineage lets you document how features are created and transformed. This adds transparency and helps you understand the relationships between features.

Read more on feature groups [4] and how to integrate them into ML systems.

4 - Next Steps: Implementing a streaming data pipeline
Imagine you’re running H&M’s online recommendation system, which delivers personalized product suggestions to millions of shoppers.

Currently, the system uses a static pipeline: embeddings for users and products are precomputed using a two-tower model and stored in an Approximate Nearest Neighbor (ANN) index.

When users interact with the site, similar products are retrieved, filtered (e.g., excluding seen or out-of-stock items), and ranked by a machine learning model.

While this approach works well offline, it struggles to adapt to real-time changes, such as shifts in user preferences or the launch of new products.

You must shift to a streaming data pipeline to make the recommendation system dynamic and responsive.


Figure 6: The Streaming Feature Pipeline in the H&M Recommender
Step 1 - Integrating real-time data
The first step is to introduce real-time data streams into your pipeline. To begin, think about the types of events your system needs to handle:

User behavior: Real-time interactions such as clicks, purchases, and searches to keep up with evolving preferences.

Product updates: Stream data on new arrivals, price changes, and stock updates to ensure recommendations reflect the most up-to-date catalog.

Embedding updates: Continuously recalculate user and product embeddings to maintain the accuracy and relevance of the recommendation model.

Step 2: Updating the retrieval stage
In a static pipeline, retrieval depends on a precomputed ANN index that matches user and item embeddings based on similarity.

However, as embeddings evolve, keeping the retrieval process synchronized with these changes is crucial to maintain accuracy and relevance.

Hopsworks supports upgrading the ANN index. This simplifies embedding updates and keeps the retrieval process aligned with the latest embeddings.

Here’s how to upgrade the retrieval stage:

Upgrade the ANN index: Switch to a system capable of incremental updates, like FAISS, ScaNN, or Milvus. These libraries support real-time similarity searches and can instantly incorporate new and updated embeddings.

Stream embedding updates: Integrate a message broker like Kafka to feed updated embeddings into the system. As a user’s preferences change or new items are added, their corresponding embeddings should be updated in real-time.

Ensure freshness: Build a mechanism to prioritize the latest embeddings during similarity searches. This ensures that recommendations are always based on the most current user preferences and available content.

Step 3: Updating the filtering stage
After retrieving a list of candidate items, the next step is filtering out irrelevant or unsuitable options. In a static pipeline, filtering relies on precomputed data like whether a user has already watched a video or if it’s regionally available.

However, filtering needs to adapt instantly to new data for a real-time system.

Here’s how to update the filtering stage:

Track recent customer activity: Use a stream processing framework like Apache Flink or Kafka Streams to maintain a real-time record of customer interactions

Dynamic stock availability: Continuously update item availability based on real-time inventory data. If an item goes out of stock, it should be filtered immediately.

Personalized filters: Apply personalized rules in real-time, such as excluding items that don’t match a customer’s size, color preferences, or browsing history.

5 - Running the feature pipeline
First, you must create an account on Hopsworks’s Serverless platform. Both making an account and running our code are free.

Then you have 3 main options to run the feature pipeline:

In a local Notebook or Google Colab: access instructions

As a Python script from the CLI, access instructions

GitHub Actions: access instructions

View the results in Hopsworks Serverless: Feature Store → Feature Groups

We recommend using GitHub Actions if you have a poor internet connection and keep getting timeout errors when loading data to Hopsworks. This happens because we push millions of items to Hopsworks.

Conclusion
In this lesson, we covered the essential components of the feature pipeline, from understanding the H&M dataset to engineering features for both retrieval and ranking models.

We also introduced Hopsworks Feature Groups, emphasizing their importance in effectively organizing, managing, and reusing features.

Lastly, we covered the transition to a real-time streaming pipeline, which is crucial for making recommendation systems adaptive to evolving user behaviors.

With this foundation, you can manage and optimize features for high-performing machine learning systems that deliver personalized, high-impact user experiences.

In Lesson 3, we’ll dive into the training pipeline, focusing on training, evaluating, and managing retrieval and ranking models using the Hopsworks model registry.

💻 Explore all the lessons and the code in our freely available GitHub repository.

If you have questions or need clarification, feel free to ask. See you in the next session!

References
Literature
[1] Decodingml. (n.d.). GitHub - decodingml/personalized-recommender-course. GitHub. https://github.com/decodingml/personalized-recommender-course

[2] Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. ACM Transactions on Information Systems, 37(1), Article 5. https://doi.org/10.1145/3285029

[3] Zheng, A., & Casari, A. (2018). Feature engineering for machine learning: Principles and techniques for data scientists. O'Reilly Media.

[4] Hopsworks. (n.d.). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/fs/feature_group/fg_overview/

[5] Hopsworks. (n.d.-b). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/fs/feature_view/fv_overview/

[6] Hopsworks. (n.d.). What is a Feature Pipeline? - Hopsworks. https://www.hopsworks.ai/dictionary/feature-pipeline

Images
If not otherwise stated, all images are created by the author.

Sponsors
Thank our sponsors for supporting our work!


Subscribe to Decoding ML
Launched 2 years ago
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Type your email...
Subscribe
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
44 Likes
∙
10 Restacks
Discussion about this post
Write a comment...
ML Educational Series
Dec 30
Edited

Setting up the repo locally but having some issues, would hhave to take a look again

TypeError: Unable to convert function return value to a Python type! The signature was

() -> handle

Like (1)
Reply
Share
ML Educational Series
Dec 31

All good 👍 here now, it was dependencies that needed to be resolved.

Like
Reply
Share

Build your Second Brain AI assistant
Using agents, RAG, LLMOps and LLM systems
Feb 6 • Paul Iusztin
870
35

LLMOps for production agentic RAG
Evaluating and monitoring LLM agents with SmolAgents and Opik
Mar 20 • Paul Iusztin and Anca Ioana Muscalagiu
94

Playbook to fine-tune and deploy LLMs
Specialized open-source LLMs for production
Mar 6 • Paul Iusztin
90
4

Ready for more?

Type your email...
Subscribe
© 2025 Paul Iusztin
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture

source 3:
Decoding ML 
Decoding ML 

Training pipelines for TikTok-like recommenders
Training ML models for a H&M real-time personalized recommender
Anca Ioana Muscalagiu
Dec 12, 2024

The third lesson of the “Hands-on H&M Real-Time Personalized Recommender” open-source course — a free course that will teach you how to build and deploy a real-time personalized recommender for H&M fashion articles using the 4-stage recommender architecture, the two-tower model design and the Hopsworks AI Lakehouse.

Lessons:
Lesson 1: Building a TikTok-like recommender

Lesson 2: Feature pipelines for TikTok-like recommenders

Lesson 3: Training pipelines for TikTok-like recommenders

Lesson 4: Deploy scalable TikTok-like recommenders

Lesson 5: Using LLMs to build TikTok-like recommenders

🔗 Learn more about the course and its outline.


Figure 1: The Training Pipeline in the FTI Architecture
Lesson 3: Training pipelines for TikTok-like recommenders
This lesson will explore the training pipeline for building and deploying effective personalized recommenders.

By the end of this lesson, you’ll learn how to:

Master the two-tower architecture

Create and use the Hopsworks retrieval Feature View effectively

Train and evaluate the two-tower network and ranking model

Upload and manage models in the Hopsworks model registry

Now, let’s dive into the "T" of the FTI Architecture by discovering how to design robust training pipelines for our real-time recommender.

💻 Explore all the lessons and the code in our freely available GitHub repository.

Table of Contents
1 - A short overview of the training pipeline components

2 - Understanding the two-tower architecture

3 - Building the training dataset for the two-tower network

4 - Training the two-tower network

5 - The ranking model: A short overview

6 - Building the ranking dataset

7 - Training and evaluating the ranking model

8 - The Hopsworks model registry

9 - Running the training pipeline

1 - A short overview of the training pipeline components
Any training pipeline inputs features and labels and outputs model(s).

📚 Read more about training pipelines and their integration into ML systems [8].

In our personalized recommender uses case, the training pipeline is composed of two key parts, each serving a distinct purpose in the recommendation workflow:

Training the two-tower network - which is responsible for narrowing down a vast catalog of items (~millions) to a smaller set of relevant candidates (~hundreds).

Training the ranking model - which refines this list of candidates by assigning relevance scores to each candidate.

As a quick reminder, check out Figure 2 for how the Customer Query Model (from the two-tower architecture) and Ranking models integrate within our H&M real-time personalized recommender.


Figure 2: The 4-stage recommender architecture applied to our H&M data
For each model, we’ll cover how to create the datasets using Hopsworks feature views, train and evaluate the models, and store them in the model registry for seamless deployment.

2 - Understanding the two-tower architecture
The two-tower architecture is the foundation of a recommender, responsible for quickly reducing a large collection of items (~ millions) to a small subset of relevant candidates (~hundreds).


Figure 3: The two-tower model, which creates the customer and article embeddings
While training the two-tower model, under the hood, it trains two models in parallel:

The Query Tower, which in our case is represented by the Customer Query Encoder.

The Candidate Tower, which in our case is represented by the Articles Encoder.

At its core, the two-tower architecture connects users and items by embedding them into a shared vector space. This allows for efficient and scalable retrieval of items for personalized recommendations.

As the name suggests, the architecture is divided into two towers:

The Query Tower: Encodes features about the user ( customer_id, age, month_sin, month_cos)

The Candidate Tower: Encodes features about the items (article_id, garment_group_name, index_group_name)

Both the Query and the Item Tower follow similar steps in embedding their respective inputs:

Feature encoding and fusion: In both models, features are preprocessed and combined. The customer_id and article_id are converted into dense embeddings, numeric values are normalized, and the categorical features are transformed using one-hot encoding.

Refinement with neural networks: A feedforward neural network with multiple dense layers refines the combined inputs into a low-dimensional embedding.


Figure 4: The two-tower architecture applied for our H&M use case
These two towers independently generate embeddings that live in the same low-dimensional space. By optimizing a similarity function (like a dot product) between these embeddings, the model learns to bring users closer to the items they are likely to interact with.

Limiting the embedding space to a low dimension is crucial to preventing overfitting. Otherwise, the model might memorize past purchases, resulting in redundant recommendations of items users already have.

We leverage the collaborative filtering paradigm by passing the customer_id and article_id as model features, which are embedded before being passed to their FNN layer.

We are pushing the network towards the content-based filtering paradigm by adding additional features to the Query Tower (age, month_sin, month_cos) and the Item Tower (garment_group_name, index_group_name). We are balancing the algorithm between the collaborative and content-based filtering paradigms depending on the number of features we add to the two-tower network (in addition to the IDs).

As an exercise, consider training the two-tower model with more features to push it to content-based filtering and see how the results change.

Having explored the two-tower architecture in depth, we continue building our model using the factory pattern.

Dig into the neural network code reflected in Figure 4 and abstracted away by the factory pattern.

query_model_factory = training.two_tower.QueryTowerFactory(dataset=dataset)
query_model = query_model_factory.build()

item_model_factory = training.two_tower.ItemTowerFactory(dataset=dataset)
item_model = item_model_factory.build()

model_factory = training.two_tower.TwoTowerFactory(dataset=dataset)
model = model_factory.build(query_model=query_model, item_model=item_model)
Read more on the two-tower architecture theory [5] or how it works with Tensorflow [2].

3 - Building the training dataset for the two-tower network
The Retrieval Feature View is the core step for preparing the training and validation datasets for the two-tower network model. Its primary role is combining user, item, and interaction data (from the feature groups) into a unified view.

We use a single dataset, the retrieval dataset, to train the query and item encoders (from the two-tower model) in parallel.

Why are Feature Views important?
Utilizing a feature view for retrieval simplifies data preparation by automating the creation of a unified dataset, ensuring:

Preventing training/inference skew: The features are ingested from a centralized repository for training and inference. Hence, we ensure they remain consistent between training and serving (aka inference).

Centralization of multiple Feature Groups: They combine features from various feature groups into a unified representation, ensuring features can easily be reused across numerous models without duplicating them, enhancing flexibility and efficiency.

For more on feature views, check out Hopsworks articles: Feature Views [3]

Defining the Feature View
To build the retrieval feature view in Hopsworks, we follow three key steps:

Get references to our Feature Groups: Reference the necessary feature groups from the Hopsworks feature store containing user-item interactions, user details, and item attributes.

trans_fg = fs.get_feature_group(name="transactions", version=1)
customers_fg = fs.get_feature_group(name="customers", version=1)
articles_fg = fs.get_feature_group(name="articles", version=1)
Select and join features: Combine relevant columns from the feature groups into a unified dataset by joining on customer_id and article_id.

selected_features = (
    trans_fg.select(["customer_id", "article_id", "t_dat", "price", "month_sin", "month_cos"])
    .join(customers_fg.select(["age", "club_member_status", "age_group"]), on="customer_id")
    .join(articles_fg.select(["garment_group_name", "index_group_name"]), on="article_id")
)
Create the Feature View: Use the unified dataset to create the retrieval feature view, consolidating all selected features into a reusable structure for training and inference.

feature_view = fs.get_or_create_feature_view(
    name="retrieval",
    query=selected_features,
    version=1,
)
This is our starting point for creating our training and validation datasets, as the train/test data split is performed directly on the Retrieval Feature View.


Figure 5: The retrieval dataset for training the two-tower network
🔗 Full code here → Github

4 - Training the two-tower network
To fit our two-tower model on the Retrieval Feature View dataset, we need to define the training step, which is composed of the following:

#1. Forward Pass & Loss computation
The forward pass computes embeddings for users and items using the Query and Item Towers. The embeddings are then used to calculate the loss:

user_embeddings = self.query_model(batch)
item_embeddings = self.item_model(batch)
loss = self.task(
   user_embeddings,
   item_embeddings,
   compute_metrics=False,
)

# Handle regularization losses as well.
regularization_loss = sum(self.losses)

total_loss = loss + regularization_loss
The metrics returned by the training step are the 3 types of losses calculated earlier:

Retrieval loss: This measures how well the model matches user embeddings to the correct item embeddings.

Regularization loss: This prevents overfitting by adding penalties for model complexity, such as large weights. It encourages the model to generalize unseen data better.

Total loss: The sum of the retrieval loss and regularization loss. It represents the overall objective the model is optimizing during training.

#2. Gradient computation & Weights updates
For gradient computation, we use Gradient Tape ( to record operations for automatic differentiation) and the AdamW optimizer, which has a configured learning rate and weight decay.

with tf.GradientTape() as tape:
  ...

  gradients = tape.gradient(total_loss, self.trainable_variables)

  self.optimizer.apply_gradients(zip(gradients,   self.trainable_variables))
Finally, we define the test step, where we perform a forward pass and calculate the loss metrics on unseen data from the test split of the dataset.

Our two-tower model is evaluated using the top-100 accuracy, where for each transaction in the validation set, the model generates a query embedding and retrieves the 100 closest items in the embedding space.

The top-100 accuracy reflects how often the actual purchased item appears within this subset, precisely measuring the model's effectiveness in retrieving relevant recommendations.

🔗 Full code here → Github

Additional Evaluation Metrics
While the methods discussed earlier are standard for evaluating recommender models, others can provide more nuanced insights into performance and user relevance. Here are some key metrics to consider:

NDCG (Normalized Discounted Cumulative Gain): A popular method for ranking that assesses both the relevance and the position of recommended items in the ranked list, giving higher importance to items placed closer to the top. Learn more about NDCG here [7].

Other Evaluation Techniques: Metrics like Precision@K (measuring the fraction of relevant items in the top-K), Recall@K (assessing coverage of relevant items), and Mean Reciprocal Rank (MRR) (indicating how quickly the first relevant item appears) provide different insights into model performance. You can learn more about diverse metrics for recommender models here [6].

5 - The ranking model: A short overview
The ranking model refines the recommendations provided by the retrieval step by assigning a relevance score to each candidate item, ensuring that the most relevant items are presented to the user.

How It Works
Model input: The ranker takes the list of filtered candidate items generated by the retrieval step (from stage 2 of the 4-stage architecture).

Model output: It predicts the likelihood of the user interacting with each candidate, assigning a relevance score to each item based on historical transaction data.

We use the CatBoostClassifier as the ranking model because it efficiently handles categorical features, requires minimal preprocessing, and provides high accuracy with built-in support for feature importance computation.

Let’s take a quick look at how to set up our dataset, model and trainer:

X_train, X_val, y_train, y_val = feature_view_ranking.train_test_split(
    test_size=settings.RANKING_DATASET_VALIDATON_SPLIT_SIZE,
    description="Ranking training dataset",
)

model = training.ranking.RankingModelFactory.build()
trainer = training.ranking.RankingModelTrainer(
    model=model, train_dataset=(X_train, y_train), eval_dataset=(X_val, y_val))
trainer.fit()
6 - Building the ranking dataset
Creating the ranking dataset is similar to the retrieval dataset, using the same Hopsworks Feature Views functionality:


Figure 6: Ranking dataset for training the ranking model.
Since the CSV files only contain positive cases (where a user purchased a specific item), we must create negative samples by pairing users with items they did not buy.

As the ratio between positive and negative samples is highly imbalanced ( 1/10 ), we use the technique of Weighted Losses, which is applied through the scale_pos_weight parameter in the CatBoostClassifier constructor:

class RankingModelFactory:
    @classmethod
    def build(cls) -> CatBoostClassifier:
        return CatBoostClassifier(
            learning_rate=settings.RANKING_LEARNING_RATE,
            iterations=settings.RANKING_ITERATIONS,
            depth=10,
            scale_pos_weight=settings.RANKING_SCALE_POS_WEIGHT,
           early_stopping_rounds=settings.RANKING_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
Full code here → Github

7 - Training and evaluating the ranking model
Training the ranking model is relatively straightforward, thanks to CatBoost's built-in fit and predict methods.

These are the key steps in the training process for the ranking model:

Dataset Preparation: The datasets are converted into CatBoost's Pool format, which efficiently handles both numerical and categorical features, ensuring the model learns effectively.

class RankingModelTrainer:
    ...

    def _initialize_dataset(self, train_dataset, eval_dataset):
        X_train, y_train = train_dataset
        X_val, y_val = eval_dataset

        cat_features = list(X_train.select_dtypes(include=["string", "object"]).columns)

        pool_train = Pool(X_train, y_train, cat_features=cat_features)
        pool_val = Pool(X_val, y_val, cat_features=cat_features)

        return pool_train, pool_val
Training Process: The fit method is used to train the model, incorporating validation data for early stopping and performance monitoring to ensure the model generalizes well.

    def fit(self):
        self._model.fit(
            self._train_dataset,
            eval_set=self._eval_dataset,
        )

        return self._model
Model Evaluation: After training, metrics like Precision, Recall, and F1-Score are calculated to assess the model’s ability to accurately rank relevant items.

   def evaluate(self, log: bool = False):
        preds = self._model.predict(self._eval_dataset)

        precision, recall, fscore, _ = precision_recall_fscore_support(
            self._y_val, preds, average="binary"
        )

        if log:
            logger.info(classification_report(self._y_val, preds))

        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }
Compute feature importance

    def get_feature_importance(self) -> dict:
        feat_to_score = {
            feature: score
            for feature, score in zip(
                self._X_train.columns,
                self._model.feature_importances_,
            )
        }

        feat_to_score = dict(
            sorted(
                feat_to_score.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return feat_to_score
Full code here → Github

8 - The Hopsworks model registry
At the end of our training pipeline, we save the trained models—the two-tower model (the Query and Candidate models) and the Ranking model—in the Hopsworks Model Registry.

Storing the models in the registry is essential. This allows them to be directly deployed in the inference pipeline of your real-time personalized recommender without extra steps.

A few key points about the Hopsworks Model Registry:

Flexible Storage: Upload your models in various formats, including Python scripts or serialized files using pickle or joblib.

Performance Tracking: Optionally includes the model schema and metrics, allowing Hopsworks to automatically identify and retrieve the best-performing version.

Seamless Deployment: Models stored in the registry integrate effortlessly with Hopsworks' serving infrastructure, making them reusable across multiple pipelines.

🔗 Full documentation → Model Registry [4]

Let’s check out a top-down approach to handling a model with Hopsworks’ Model Registry by digging into the HopsworksQueryModel class.

query_model = hopsworks_integration.two_tower_serving.HopsworksQueryModel(
    model=model.query_model
)
query_model.register(
    mr=mr,
    query_df=dataset.properties["query_df"],
    emb_dim=settings.TWO_TOWER_MODEL_EMBEDDING_SIZE,
)
The HopsworksQueryModel class integrates the Hopsworks’ model registry to handle the query tower of the two-tower model.

To save our trained model in Hopsworks, we use the register() method, which can be broken down into the following steps:

Save the model locally:

class HopsworksQueryModel:
    ... # More code

    def register(self, mr, query_df, emb_dim) -> None:
        local_model_path = self.save_to_local()
Extract an input example from the DataFrame:

        query_example = query_df.sample().to_dict("records")
Create the Tensorflow Model using the model schema, input example:

        mr_query_model = mr.tensorflow.create_model(
            name="query_model",  # Name of the model
            description="Model that generates query embeddings from user and transaction features",  # Description of the model
            input_example=query_example,  # Example input for the model
            feature_view=feature_view, # Model's input feature view.
        )
Important: By attaching the feature view used to train the model, we can retrieve the exact version of the feature view the model was trained on at serving time. Thus, when the model is deployed, we guarantee that the right features will be used for inference. Eliminating one aspect of the training-inference skew.

Save the model to the model registry:

        mr_query_model.save(local_model_path)  # Path to save the model
Once the model is registered, it can be easily retrieved at inference time together with its feature view using the Hopsworks API:

# Retrieve the 'query_model' from the Model Registry
query_model = mr.get_model(
   name="query_model",
   version=1,
)

# Retrieve the 'query_feature_view' used to train the model
query_fv = query_model.get_feature_view(init=False) 
🔗 Full code here → Github

9 - Running the training pipeline
Now that we’ve covered each model in detail. Let's take a step back to review the retrieval and ranking training pipelines:


Figure 7: Overview of the Training Pipeline
To run the training pipelines, follow the following steps:

First, you must create an account on Hopsworks’s Serverless platform. Both making an account and running our code are free.

Next, you must run the feature pipeline to populate the Hopsworks feature groups. Afterward, you can run the training pipeline to train the models.

To run the feature and training pipelines, you have 3 options:

In a local Notebook or Google Colab: access instructions

As a Python script from the CLI, access instructions

GitHub Actions: access instructions

View the results in Hopsworks Serverless: Data Science → Model Registry

We recommend using GitHub Actions if you have a poor internet connection and keep getting timeout errors when loading data to Hopsworks. This happens because we push millions of items to Hopsworks.

Conclusion
This lesson taught us about the two-tower architecture, feature views, the query encoder, the candidate encoder and the ranking model.

Then, we explored how to train and evaluate these models while leveraging the features computed in the feature pipeline from Hopsworks feature views.

Lastly, we saw how to upload, version and share our trained models through the Hopsworks model registry.

In Lesson 4, we will continue with the inference pipeline, digging into implementing the 4-stage pattern and deploying it with Hopsworks and KServe.

💻 Explore all the lessons and the code in our freely available GitHub repository.

If you have questions or need clarification, feel free to ask. See you in the next session!

References
Literature
[1] Decodingml. (n.d.). GitHub - decodingml/personalized-recommender-course. GitHub. https://github.com/decodingml/personalized-recommender-course

[2] Wortz, J., & Totten, J. (2023, April 19). Tensorflow deep retrieval using Two Towers architecture. Google Cloud Blog. https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture

[3] Hopsworks. (n.d.-b). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/fs/feature_view/fv_overview/

[4] Hopsworks. (n.d.). Overview - HopsWorks documentation. https://docs.hopsworks.ai/latest/concepts/mlops/registry/

[5] Hopsworks. (n.d.). What is a Two-Tower Embedding Model? - Hopsworks. https://www.hopsworks.ai/dictionary/two-tower-embedding-model

[6] 10 metrics to evaluate recommender and ranking systems. (n.d.). https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems

[7] Normalized Discounted Cumulative Gain (NDCG) explained. (n.d.). https://www.evidentlyai.com/ranking-metrics/ndcg-metric

[8] Hopsworks. (n.d.-b). What is a Training Pipeline? - Hopsworks. https://www.hopsworks.ai/dictionary/training-pipeline

Images
If not otherwise stated, all images are created by the author.

Sponsors
Thank our sponsors for supporting our work!


Subscribe to Decoding ML
Launched 2 years ago
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Type your email...
Subscribe
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
27 Likes
∙
10 Restacks
Discussion about this post
Write a comment...
Cha_le
Dec 25

Hi, Thank you for the blog post and the tutorial.

I have some questions on the ranking model. In the example we use cat boost and aim for 1 and 0 label on if the customer buy the product or not.

Aren't the result of this model will just be the prediction of the customer likely to buy the article or not?

This model will not output the recommendation in ranking, right?

So in my mind, we can only use it to filter out products customers are not gonna buy. But we still don't really have an ordering result of which product to show first.

Maybe I am missing something. If anyone can help me verify this that would be very helpful

Thank you

Like (1)
Reply
Share
6 replies by Paul Iusztin and others
Sitraka FORLER
Jan 17

Wow really cleaar...and impressive to know !

Thanks a lot for sharing it!!!!

Like
Reply
Share
6 more comments...

Build your Second Brain AI assistant
Using agents, RAG, LLMOps and LLM systems
Feb 6 • Paul Iusztin
870
35

LLMOps for production agentic RAG
Evaluating and monitoring LLM agents with SmolAgents and Opik
Mar 20 • Paul Iusztin and Anca Ioana Muscalagiu
94

Playbook to fine-tune and deploy LLMs
Specialized open-source LLMs for production
Mar 6 • Paul Iusztin
90
4

Ready for more?

Type your email...
Subscribe
© 2025 Paul Iusztin
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture


source 4:
Decoding ML 
Decoding ML 

Deploy scalable TikTok-like recommenders
Ship to the real world an H&M recommender using KServe
Paul Iusztin
Dec 26, 2024

The fourth lesson of the “Hands-on H&M Real-Time Personalized Recommender” open-source course — a free course that will teach you how to build and deploy a real-time personalized recommender for H&M fashion articles using the 4-stage recommender architecture, the two-tower model design and the Hopsworks AI Lakehouse.

Lessons:
Lesson 1: Building a TikTok-like recommender

Lesson 2: Feature pipelines for TikTok-like recommenders

Lesson 3: Training pipelines for TikTok-like recommenders

Lesson 4: Deploy scalable TikTok-like recommenders

Lesson 5: Using LLMs to build TikTok-like recommenders

🔗 Learn more about the course and its outline.


Figure 1: The inference pipeline in the FTI architecture
Lesson 4: Deploy scalable TikTok-like recommenders
This lesson will wrap up our H&M personalized recommender project by implementing and deploying the inference pipelines of our ML system, as illustrated in Figure 1.

Serving ML models is one of the most complex steps when it comes to AI/ML in production, as you have to put all the pieces together into a unified system while considering:

throughput/latency requirements

infrastructure costs

data and model access

training-serving skew

As we started this project with production in mind by using the Hopsworks AI Lakehouse, we can easily bypass most of these issues, such as:

the query and ranking models are accessed from the model registry;

the customer and H&M article features are accessed from the feature store using the offline and online stores depending on throughput/latency requirements;

the features are accessed from a single source of truth (feature store), solving the training-serving skew.

Estimating infrastructure costs in a PoC is more complicated. Still, we will leverage a Kubernetes cluster managed by Hopsworks, which uses KServe to scale up and down our real-time personalized recommender depending on traffic.

Thus, in this lesson, you will learn how to:

Architect offline and online inference pipelines using MLOps best practices.

Implement offline and online pipelines for an H&M real-time personalized recommender.

Deploy the online inference pipeline using the KServe engine.

Test the H&M personalized recommender from a Streamlit app.

Deploy the offline ML pipelines using GitHub Actions.

Table of Contents:
Understanding the architecture of the inference pipelines

Building the offline candidate embedding inference pipeline

Implementing the online query service

Implementing the online ranking service

Deploying the online inference pipelines using KServe

Testing the H&M real-time personalized recommender

Deploying the offline ML pipelines using GitHub Actions

1 - Understanding the architecture of the inference pipelines
Before going into the implementation details, we want to explain the serving strategy of our inference pipelines. We have one offline and one online inference pipeline.

Here is a quick refresher on inference pipelines [2].

Let’s understand the difference between the two in our personalized recommender.

Online vs. offline inference pipelines in recommenders
The inference pipeline is split into 2 main processes to optimize for real-time recommendations.

The offline pipeline runs in batch mode, optimized for high throughput. It embeds all the candidate items from our database using the candidate articles encoder (trained using the two-tower network).

The offline pipeline runs once to backfill our H&M articles collection. It should then run again whenever a new article is added to our collection or the two-tower network is retrained (which changes our embedding space).


Figure 2: End-to-end architecture of the H&M real-time personalized recommender
The online inference pipeline is deployed as a real-time service optimized for low latency. It will run on each client request, serving e-commerce personalized recommendations to each client.


Figure 3: The two-tower network: Create customer and article embeddings in the same vector space
Now, let’s zoom in on each pipeline.

Offline inference pipeline
The offline pipeline loads the article candidate encoder from the Hopsworks model registry and a reference to the retrieval feature view from the Hopsworks feature store.

Leveraging the feature view, it feeds in all the necessary features to the encoder, avoiding any potential training-serving skew.


Figure 4: Understanding how the offline pipeline connects to the online pipeline
Ultimately, it saves the candidate embeddings into a new feature group that supports a vector index for semantic search between the H&M fashion items and the user query. We flag the feature group as online to be optimized for low latency requirements.

We create a feature view from the feature group to expose the embedding vector index to the online inference pipeline.

Important! Behind-the-scenes insights
Labeling components in ML systems is hard!

For example, we labeled the candidate embeddings pipeline an “inference pipeline” because we examined its inputs: a trained model from the model registry and input features from the feature store.

However, based on Jim Downling’s feedback (CEO of Hopsworks), a way to consistently label your pipelines is based on the ML artifact/asset they produce.

Thus, if we look at its outputs, embeddings written to a feature group are used as features in downstream pipelines… We should have labeled it as a “feature pipeline.”

Engineers constantly struggle with labeling components in software systems.

However, consistency across the system is essential. That’s why Jim’s approach of labeling each pipeline according to the ML asset it produces is intuitive and a strong strategy to consider!

Online inference pipeline
The online inference pipeline implements the 4-stage architecture, which we kept talking about throughout this course.

The problem with real-time recommenders is that you must narrow from millions to dozens of item candidates in less than a second while the items are personalized to the user.

The 4-stage recommender architecture solves that!


Figure 5: The 4-stage recommender system applied to our H&M data
Here is a quick reminder of the 4 stages we have to implement:

Stage 1: Take the customer_id and other input features, such as the current date, compute the customer embedding using the Customer Query Model and query the Hopsworks vector DB for similar candidate items — Reduce a corpus of millions of items to ~hundreds.

Stage 2: Takes the candidate items and applies various filters, such as removing items already seen or purchased using a Bloom filter.

Stage 3: During ranking, we load more features from Hopsworks' feature store describing the item and the user's relationship: "(item candidate, customer)." This is feasible as only a few hundred items are being scored, compared to the millions scored in candidate generation. The ranking model can use a boosting tree, such as XGBoost or CatBoost, a neural network or even an LLM.

Stage 4: We order the items based on the ranking score plus other optional business logic. The highest-scoring items are presented to the user and ranked by their score — Redice the ~hundreds of candidates of items to ~dozens.

All these recommendations are computed in near real-time (in milliseconds).

More on the 4 stage architecture in the first lesson:

Building a TikTok-like recommender
Building a TikTok-like recommender
Paul Iusztin
·
November 28, 2024
Read full story
Serving real-time recommendations using Hopsworks Serverless and KServe
We will deploy the online inference pipeline to Hopsworks Serverless, which uses KServe under the hood to serve the models.

What is KServe? It’s a runtime engine designed to serve predictive and generative ML models on Kubernetes clusters. It streamlines the complexities of autoscaling, networking, health checks, and server configuration, offering advanced serving features such as GPU autoscaling, scaling to zero, and canary rollouts for your ML deployments. 🔗 More on KServe [3]


Figure 6: Deploying the online inference pipeline to Hopsworks Serverless using KServe
Leveraging KServe, we will deploy two different services:

The query encoder service

The ranking service

Why?

We deploy them as two services because each has its model and environment. Thus, following KServe’s best practices, we will wrap each model into its own Predictor, which can be scaled and optimized independently.

The Transformer component is used to preprocess and postprocess the results from the Predictor (aka the model).

…and no! It has nothing to do with LLM — Transformer architectures. Not anything revolves around LLMs!

The KServe flow will be as follows:

The client calls the query service and sends its ID and transaction date.

The query service preprocesses the request within the Transformer (such as calling the feature store to get the client’s features based on its ID).

The query service calls the customer encoder Predictor.

The query service calls the ranking service, passing the query embedding.

The ranking service preprocesses the request within its Transformer, calls the ranking model and post-processes the recommendations.

The ranking service sends the results to the query service, which then sends the results back to the client.

Let’s dig into the code to see how this works in practice while using Hopsworks AI Lakehouse to power the ML system.

2 - Building the offline candidate embedding inference pipeline
The first step is to run our offline candidate embedding inference pipeline (in batch mode) to populate our Hopsworks vector index with all our H&M article embeddings.

Here is the implementation:

We connect to Hopsworks, our feature store and model registry platform. From there, we download our previously trained candidate model (within the two-tower network), which we'll use to generate item embeddings:

from recsys import features, hopsworks_integration
from recsys.config import settings


project, fs = hopsworks_integration.get_feature_store()
mr = project.get_model_registry()

candidate_model, candidate_features = (
    hopsworks_integration.two_tower_serving.HopsworksCandidateModel.download(mr=mr)
)
Next, we fetch our data using the retrieval feature view. The benefit of using a feature view is that the data already contains all the necessary features for our item embeddings. Thus, following the FTI architecture, no feature engineering is required at this point:

feature_view = fs.get_feature_view(
    name="retrieval",
    version=1,
)

train_df, val_df, test_df, _, _, _ = feature_view.train_validation_test_split(
    validation_size=settings.TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE,
    test_size=settings.TWO_TOWER_DATASET_TEST_SPLIT_SIZE,
    description="Retrieval dataset splits",
)

Figure 7: Example of the retrieval feature view in Hopsworks Serverless.
The core step of the offline inference pipeline is to take the item features and the candidate model and compute all the embeddings in batch mode:

item_df = features.embeddings.preprocess(train_df, candidate_features)
embeddings_df = features.embeddings.embed(df=item_df, candidate_model=candidate_model)
The preprocess() isn’t performing any feature engineering but just dropping any potential article duplicates:

item_df.drop_duplicates(subset="article_id", inplace=True)
Within the embed() function, we call the embedding model in batch mode while transforming the results into a Pandas DataFrame containing the article IDs and embeddings. The ID is critical in identifying the article after retrieving the candidates using semantic search:

def embed(df: pd.DataFrame, candidate_model) -> pd.DataFrame:
    ds = tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})

    candidate_embeddings = ds.batch(2048).map(
        lambda x: (x["article_id"], candidate_model(x))
    )

    all_article_ids = tf.concat([batch[0] for batch in candidate_embeddings], axis=0)
    all_embeddings = tf.concat([batch[1] for batch in candidate_embeddings], axis=0)

    all_article_ids = all_article_ids.numpy().astype(int).tolist()
    all_embeddings = all_embeddings.numpy().tolist()

    embeddings_df = pd.DataFrame(
        {
            "article_id": all_article_ids,
            "embeddings": all_embeddings,
        }
    )

    return embeddings_df
We store these embeddings in Hopsworks by creating a dedicated feature group with an embedding index. By enabling online access, we ensure these embeddings will be readily available for our real-time recommendation service:

candidate_embeddings_fg = create_candidate_embeddings_feature_group(
        fs=fs, df=embeddings_df, online_enabled=True
)

Figure 8: Example of the candidate embeddings feature group in Hopsworks Serverless.
Ultimately, we create a feature view based on the embeddings feature group to expose the vector index to the online inference pipeline:

feature_view = create_candidate_embeddings_feature_view(
        fs=fs, fg=candidate_embeddings_fg
)

Figure 9: Previewing the ingested candidate embeddings in Hopsworks Serverless
Full Notebook and code are available on our GitHub.

3 - Implementing the online query service
Now that the vector index is populated with H&M fashion article candidate embeddings, we will focus on building our recommender online inference pipeline, which implements the 4-stage architecture.

We must implement a class following the Transformer interface, as we use KServe to deploy our query and ranking models.

The flow of the Transformer class is as follows:

Calls the preprocess() method to prepare the data before feeding it to the model.

Calls the deployed model (in our case, the Query encoder model)

Calls the postprocess() method to process the data before returning it to the client.


Figure 6: Deploying the online inference pipeline to Hopsworks Serverless using KServe
Now, let’s dig into the implementation:

First, we define the Transformer class and get references to the ranking feature view (used to train the two-tower network) and the ranking KServe deployment. We need a reference to the ranking service as we have to pass it the query embedding to complete the steps from the 4-stage architecture:

from datetime import datetime

import hopsworks
import numpy as np
import pandas as pd


class Transformer(object):
    def __init__(self) -> None:
        project = hopsworks.login()
        ms = project.get_model_serving()
        fs = project.get_feature_store()

        self.customer_fv = fs.get_feature_view(
            name="customers",
            version=1,
        )
        self.ranking_fv = fs.get_feature_view(name="ranking", version=1)
        self.ranking_fv.init_batch_scoring(1)

        # Retrieve the ranking deployment
        self.ranking_server = ms.get_deployment("ranking")
The preprocessing logic transforms raw API inputs into model-ready features. Note how we leveraged the Hopsworks feature view to ensure the features are consistent and computed the right way during inference to avoid the training-serving skew (for both static and on-demand features):

    def preprocess(self, inputs):
        customer_id = inputs["customer_id"]
        transaction_date = inputs["transaction_date"]
        month_of_purchase = datetime.fromisoformat(inputs.pop("transaction_date"))

        # Real-time feature serving from the feature store
        customer_features = self.customer_fv.get_feature_vector(
            {"customer_id": customer_id},
            return_type="pandas",
        )
        inputs["age"] = customer_features.age.values[0]

        # Use the feature view for on-demand feature computation to avoid train-serving skew.
        feature_vector = self.ranking_fv._batch_scoring_server.compute_on_demand_features(
            feature_vectors=pd.DataFrame([inputs]), 
            request_parameters={"month": month_of_purchase}
        ).to_dict(orient="records")[0]

        inputs["month_sin"] = feature_vector["month_sin"]
        inputs["month_cos"] = feature_vector["month_cos"]

        return {"instances": [inputs]}
The postprocessing step is straightforward - it takes the model's raw predictions and uses our ranking server to generate the final ordered recommendations:

    def postprocess(self, outputs):
        return self.ranking_server.predict(inputs=outputs)
Note that the KServe runtime within the Predictor component implicitly calls the Query encoder model. Still, we must explicitly upload the model when deploying our service, which we will show you later in this article.

We have only implemented Step 1 of the 4-stage architecture so far. The rest will be in the ranking service.

The complete Transformer class is available on our GitHub.

4 - Implementing the online ranking service
The last piece of our online inference pipeline is the ranking service, which communicates directly with the query service, as we saw in its postprocess() method.

As with the Query encoder, we have to implement the Transformer interface:

We initialize all the required features to perform the rest of the steps from the 4-stage architecture. One powerful feature of Hopsworks is that it allows us to automatically grab the feature view (along with its version) on which the ranking model was trained, eliminating another training-serving skew scenario:

class Transformer(object):
    def __init__(self):
        # Connect to Hopsworks
        project = hopsworks.login()
        self.fs = project.get_feature_store()
        
        # Get feature views and groups
        self.transactions_fg = self.fs.get_feature_group("transactions", 1)
        self.articles_fv = self.fs.get_feature_view("articles", 1)
        self.customer_fv = self.fs.get_feature_view("customers", 1)
        self.candidate_index = self.fs.get_feature_view("candidate_embeddings", 1)
        
        # Initialize serving
        self.customer_fv.init_serving(1)
        
        # Get ranking model and features
        mr = project.get_model_registry()
        model = mr.get_model(name="ranking_model", version=1)
        self.ranking_fv = model.get_feature_view(init=False)
        self.ranking_fv.init_batch_scoring(1)
The preprocessing stage is where the real magic happens. When a request comes in, we first retrieve candidate items using vector similarity search based on the customer's query embedding, computed by the Query KServe service. We then filter out items the customer has already purchased by checking the transactions feature group, which is part of Stage 2:

def preprocess(self, inputs):
    customer_id = inputs["instances"][0]["customer_id"]
    
    # Get and filter candidates
    neighbors = self.candidate_index.find_neighbors(
        inputs["query_emb"],
        k=100,
    )
    neighbors = [neighbor[0] for neighbor in neighbors]
    
    already_bought_items_ids = (
        self.transactions_fg.select("article_id")
        .filter(self.transactions_fg.customer_id==customer_id)
        .read(dataframe_type="pandas").values.reshape(-1).tolist()
    )
    
    item_id_list = [
        str(item_id)
        for item_id in neighbors
        if str(item_id) not in already_bought_items_ids
    ]
Next, we move on to Stage 3, where we enrich our candidates with features from the articles and customer feature views. We combine article features, customer demographics, and temporal features (month sine/cosine) to create a richer feature spectrum leveraged by the ranking model to understand better how relevant an H&M item is to the user:

    # Get article and customer features
    articles_data = [
        self.articles_fv.get_feature_vector({"article_id": item_id})
        for item_id in item_id_list
    ]
    articles_df = pd.DataFrame(data=articles_data, columns=self.articles_features)
    
    customer_features = self.customer_fv.get_feature_vector(
        {"customer_id": customer_id},
        return_type="pandas",
    )
    
    # Combine all features
    ranking_model_inputs = item_id_df.merge(articles_df, on="article_id", how="inner")
    ranking_model_inputs["age"] = customer_features.age.values[0]
    ranking_model_inputs["month_sin"] = inputs["month_sin"]
    ranking_model_inputs["month_cos"] = inputs["month_cos"]
Finally, after the ranking model scores the candidates, we move to Stage 4 and sort the articles, representing our final ordered recommendations. This is our final step, providing a ranked list of personalized product recommendations to the user:

def postprocess(self, outputs):
    ranking = list(zip(outputs["scores"], outputs["article_ids"]))
    ranking.sort(reverse=True)

    return {"ranking": ranking}
The complete Transformer class is available on our GitHub.

As before, the ranking model is implicitly called between the preprocess() and postprocess() methods. But there is a catch…

As we use CatBoost as our ranking module, KServe doesn’t know how to load it out-of-the-box, as it happened for the Tenforflow/Keras Query encoder.

Thus, similar to the Transformer interface, we must implement the Predictor interface explicitly defining how the model is loaded and called. This interface is much more straightforward as we must implement a single predict() method. Let’s take a look:

Define the class and the __init__ method, where we load the CatBoost model:

class Predict(object):
    def __init__(self):
        self.model = joblib.load(os.environ["MODEL_FILES_PATH"] + "/ranking_model.pkl")
The core prediction logic happens in the predict() method, which is called by KServe's inference service. First, we extract the ranking features and article IDs from the input payload. Our transformer component previously prepared these features:

    def predict(self, inputs):
        features = inputs[0].pop("ranking_features")
        article_ids = inputs[0].pop("article_ids")
The final step is where the actual ranking happens. We use our loaded model to predict probabilities for each candidate article, focusing on the positive class scores. The scores are paired with their corresponding article IDs in the response:

        scores = self.model.predict_proba(features).tolist()
        scores = np.asarray(scores)[:,1].tolist() 

        return {
            "scores": scores, 
            "article_ids": article_ids,
        }
The predictor integrates with KServe's inference pipeline alongside the transformer component that handles feature preprocessing. This setup allows us to serve real-time recommendations through a scalable Kubernetes infrastructure.

The complete Predict class is available on our GitHub.

5 - Deploying the online inference pipelines using KServe
Now that we have our fine-tuned models and Transformer & Predict classes in place, the last step is to ship them to a Kubernetes cluster managed by Hopsworks Serverless using KServe.

Hopsworks makes this easy. Let’s see how it works:

Let's start with our environment setup and Hopsworks connection:

import warnings
warnings.filterwarnings("ignore")

from loguru import logger
from recsys import hopsworks_integration

project, fs = hopsworks_integration.get_feature_store()
We first deploy our ranking model to Hopsworks Serveless, leveraging our custom HopsworksRankingModel Python class.

ranking_deployment = ranking_serving.HopsworksRankingModel.deploy(project)

ranking_deployment.start()
Behind the scenes, the deployment method uploads the necessary transformer and predictor scripts to Hopsworks, selects the best-ranking model from the model registry based on the F-score metric, and configures a KServe transformer for preprocessing.

Initially, we configure the deployment with zero instances, autoscaling based on demand. We want to let the demo run 24/7. Thus, we can save tons on costs by setting the instances to 0 when there is no traffic. Hopsworks serverless takes care of autoscaling out-of-the-box:

from hsml.transformer import Transformer

from recsys.config import settings


class HopsworksRankingModel:
    deployment_name = "ranking"

    ... # Other methods

    @classmethod
    def deploy(cls, project):
        mr = project.get_model_registry()
        dataset_api = project.get_dataset_api()

        ranking_model = mr.get_best_model(
            name="ranking_model",
            metric="fscore",
            direction="max",
        )

        # Copy transformer file into Hopsworks File System
        uploaded_file_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "inference" / "ranking_transformer.py"),
            "Resources",
            overwrite=True,
        )
        transformer_script_path = os.path.join(
            "/Projects",  # Root directory for projects in Hopsworks
            project.name,
            uploaded_file_path,
        )

        # Upload predictor file to Hopsworks
        uploaded_file_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "inference" / "ranking_predictor.py"),
            "Resources",
            overwrite=True,
        )
        predictor_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path,
        )

        ranking_transformer = Transformer(
            script_file=transformer_script_path,
            resources={"num_instances": 0},
        )

        # Deploy ranking model
        ranking_deployment = ranking_model.deploy(
            name=cls.deployment_name,
            description="Deployment that search for item candidates and scores them based on customer metadata",
            script_file=predictor_script_path,
            resources={"num_instances": 0},
            transformer=ranking_transformer,
        )

        return ranking_deployment
The complete class code is available on GitHub.

For testing the ranking deployment, we use a sample input that matches our transformer's expected format:

def get_top_recommendations(ranked_candidates, k=3):
    return [candidate[-1] for candidate in ranked_candidates["ranking"][:k]]

test_ranking_input = [
    {
        "customer_id": "d327d0ad9e30085a436933dfbb7f77cf42e38447993a078ed35d93e3fd350ecf",
        "month_sin": 1.2246467991473532e-16,
        "query_emb": [0.214135289, 0.571055949, /* ... */],
        "month_cos": -1.0,
    }
]

ranked_candidates = ranking_deployment.predict(inputs=test_ranking_input)
recommendations = get_top_recommendations(ranked_candidates["predictions"], k=3)
For the Query encoder model, we follow a similar strategy:

query_model_deployment = two_tower_serving.HopsworksQueryModel.deploy(project)

query_model_deployment.start()
Under the hood, the deploy() method is similar to the one from the HopsworksRankingModel class:


from recsys.config import settings
from recsys.training.two_tower import ItemTower, QueryTower


class HopsworksQueryModel:
    deployment_name = "query"

    ... # Other methods

    @classmethod
    def deploy(cls, project):
         ... # Similar code to the ranking model

         query_model_deployment = query_model.deploy(
            name=cls.deployment_name,
            description="Deployment that generates query embeddings.",
            resources={"num_instances": 0},
            transformer=query_model_transformer,
        )

        return query_model_deployment
The complete class code is available on GitHub.

Testing the query model requires only the customer_id and transaction_date, as the transformer handles taking all the required features from Hopsworks feature views, avoiding any state transfer between the client and ML service:

data = [
    {
        "customer_id": "d327d0ad9e30085a436933dfbb7f77cf42e38447993a078ed35d93e3fd350ecf",
        "transaction_date": "2022-11-15T12:16:25.330916",
    }
]

ranked_candidates = query_model_deployment.predict(inputs=data)
recommendations = get_top_recommendations(ranked_candidates["predictions"], k=3)
Finally, we clean up our resources:

ranking_deployment.stop()
query_model_deployment.stop()
After running the deployment steps, you should see them in Hopsworks Serverless, as Figure 10 illustrates under the Data Science → Deployments section.


Figure 10: View results in Hopsworks Serverless: Data Science → Deployments
The deployment logic is not dependent on Hopsworks.

Even if we used a managed version of Kubernetes + KServe on Hopsworks Serverless to deploy our inference pipelines, you could leverage the same code (Transformer and Predictor classes) and trained models on any other KServe infrastructure.

Full Notebook and code are available on our GitHub.

6 - Testing the H&M real-time personalized recommender
We are finally here: Where we can test our H&M real-time personalized recommender!

For testing the online inference pipeline, we wrote a simple Streamlit app that allows you to visualize the real-time recommendations for different users and generate new interactions to adapt future recommendations.


Figure 11: Example of the Streamlit app.
We won’t get into the Streamlit code, but under the hood, calling the real-time deployment through Hopsworks is as easy as:

project, fs = hopsworks_integration.get_feature_store()
ms = project.get_model_serving()

query_model_deployment = ms.get_deployment(
        HopsworksQueryModel.deployment_name
)
query_model_deployment.start(await_running=180)


deployment_input = [
                {
     "customer_id": customer_id, 
     "transaction_date": formatted_timestamp}
]

prediction = query_model_deployment.predict(inputs=deployment_input)[
                "predictions"
            ]["ranking"]
Beautiful, right?

Everything else is Streamlit code!

Which you can find in our GitHub repository.

Running the code
Assuming you finalized the feature engineering and training steps explained in previous lessons, you can generate the embeddings by running:

make create-embeddings
View results in Hopsworks Serverless → Feature Store → Feature Groups

Then, you can create the deployments by running:

make create-deployments
View results in Hopsworks Serverless → Data Science → Deployments

Ultimately, you can start the Streamlit app as follows — Accessible at `http://localhost:8501/`:

make start-ui
🌐 We also deployed a live demo to play around with the H&M personalized recommender effortlessly: Live demo ←

The first time you interact with the demo, it will take a while to warm up the deployment from 0 to +1 instances. After that, the deployments will happen in real-time. This happens because we are in demo, 0-cost mode, scaling to 0 instances when there is no traffic.

Step-by-step-instructions
For the complete guide, access the GitHub documentation.

Step-by-step instructions for running the code:

In a local Notebook or Google Colab: access instructions

As a Python script from the CLI, access instructions

GitHub Actions: access instructions

Deploy the Streamlit app: access instructions

We recommend using GitHub Actions if you have a poor internet connection and keep getting timeout errors when loading data to Hopsworks. This happens because we push millions of items to Hopsworks.

7 - Deploying the offline ML pipelines using GitHub Actions
GitHub Actions is a great way to deploy offline ML pipelines that don’t require much computing power.

Why? When working with public repositories, they are free and can easily be integrated with your code.

As shown in Figure 12, we can easily chain multiple Python programs within a DAG. For example, after the features are successfully computed, we can leverage more complex relationships by running both training pipelines in parallel.


Figure 12: Example of the GitHub Actions flow running the offline ML pipeline. Access our examples ←
As we work with a static H&M dataset, we should run our offline ML pipelines only once to backfill our feature store, as our features, models and candidate embeddings don’t change. Still, in a real-world scenario, our data won’t be static, and we could easily leverage GitHub Actions to do continuous training once the code changes or new data is available.

Another massive benefit of using GitHub Actions is that it provides enterprise-level network access, saving you tons of headaches when working with medium to large datasets that can easily throw network errors on more unstable home Wi-Fis.

This can also happen in our H&M use case, where we work with millions of samples when loading the features to Hopsworks.

Now, let’s quickly dive into the GitHub Actions implementation:

We can run the pipeline automatically on a schedule (every 2 hours), on code changes, or manually through GitHub's UI. The pipeline takes approximately 1.5 hours to complete, which influenced these timing choices:

name: ML Pipelines

on:
  # schedule: # Run pipelines every 2 hours
  #   - cron: '0 */2 * * *'
  # push: # Run on every new commit to main
  #   branches:
  #     - main
  workflow_dispatch:  # Manual triggering
      
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
The pipeline begins with feature engineering:

jobs:
  feature_engineering:
    name: Compute Features
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
      - uses: ./.github/actions/setup
      - name: Run pipeline
        run: uv run ipython notebooks/1_fp_computing_features.ipynb
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
Once features are ready, the pipeline branches into parallel training jobs for two distinct models: the retrieval model and the ranking model:

  train_retrieval:
    needs: feature_engineering
    name: Train Retrieval Model
    # ... similar setup steps ...

  train_ranking:
    needs: feature_engineering
    name: Train Ranking Model
    # ... similar setup steps ...
After the retrieval model training completes, we compute and index item embeddings:

  computing_and_indexing_embeddings:
    needs: train_retrieval
    name: Compute Embeddings
    # ... similar setup steps ...
The final step creates the deployments:

  create_deployments:
    needs: computing_and_indexing_embeddings
    name: Create Deployments
    # ... similar setup steps ...
As you can see, deploying and running our offline ML pipeline through GitHub Actions while leveraging free computing is easy.

See our GitHub Actions runs or the complete code.

Conclusion
Congratulations! After finishing this lesson, you created an end-to-end H&M real-time personalized recommender.

Within this lesson, you learned how to architect, implement and deploy offline and online inference pipelines using the Hopsworks AI Lakehouse.

Also, you’ve learned how to test the personalized recommender from a Streamlit app, highlighting how easy it is to leverage Hopsworks SDK for real-time ML deployments.

Ultimately, as a bonus, you’ve learned how to deploy and schedule all the offline ML pipelines using GitHub Actions.

Even if we finished the H&M personalized recommender, we are not done with the course yet!

In Lesson 5, we prepared something exciting: We will learn to enhance our H&M personalized recommender with LLMs.

💻 Explore all the lessons and the code in our freely available GitHub repository.

If you have questions or need clarification, feel free to ask. See you in the next session!

References
Literature
[1] Decodingml. (n.d.). GitHub - decodingml/personalized-recommender-course. GitHub. https://github.com/decodingml/personalized-recommender-course

[2] Hopsworks. (n.d.). What is an Inference Pipeline? - Hopsworks. https://www.hopsworks.ai/dictionary/inference-pipeline

[3] Hopsworks. (n.d.). What is Kserve? - Hopsworks. https://www.hopsworks.ai/dictionary/kserve

Images
If not otherwise stated, all images are created by the author.

Sponsors
Thank our sponsors for supporting our work!


Subscribe to Decoding ML
Launched 2 years ago
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Type your email...
Subscribe
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
24 Likes
∙
8 Restacks
Discussion about this post
Write a comment...
ML Educational Series
Dec 29

Wonderful work there Paul, I did a quick read, am yet to go through the implementation the fun part for me. Kudos Paul

Like (1)
Reply
Share
1 reply by Paul Iusztin
1 more comment...

Build your Second Brain AI assistant
Using agents, RAG, LLMOps and LLM systems
Feb 6 • Paul Iusztin
870
35

LLMOps for production agentic RAG
Evaluating and monitoring LLM agents with SmolAgents and Opik
Mar 20 • Paul Iusztin and Anca Ioana Muscalagiu
94

Playbook to fine-tune and deploy LLMs
Specialized open-source LLMs for production
Mar 6 • Paul Iusztin
90
4

Ready for more?

Type your email...
Subscribe
© 2025 Paul Iusztin
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture

source 5:
Decoding ML 
Decoding ML 

Using LLMs to build TikTok-like recommenders
How LLMs transform classic recommender architectures
Paolo Perrone
Jan 09, 2025

The fifth and final lesson of the “Hands-On Real-time Personalized Recommender” open-source course — a free course that will teach you how to build and deploy a production-ready real-time personalized recommender for H&M articles using the four-stage recsys architecture, the two-tower model design and the Hopsworks AI Lakehouse.

Lessons:
Lesson 1: Building a TikTok-like recommender

Lesson 2: Feature pipelines for TikTok-like recommenders

Lesson 3: Training pipelines for TikTok-like recommenders

Lesson 4: Deploy scalable TikTok-like recommenders

Lesson 5: Using LLMs to build TikTok-like recommenders

🔗 Learn more about the course and its outline.


Figure 1: The Inference pipeline in the FTI architecture (powered by LLMs)
Lesson 5: Using LLMs to build TikTok-like recommenders
In our previous lessons, we built and deployed a production-ready H&M personalized recommender using classical machine learning techniques.

We used a two-tower architecture for candidate generation and leveraged CatBoost for ranking, creating a scalable system for real-time recommendations.

Now, let’s enhance our recommender system by incorporating LLMs.

By the end of this lesson, you will learn how to:

Use LLMs for ranking in recommendation systems

Deploy the LLM-based ranking service using KServe

Use semantic search for personalized recommendations

Evaluate the effectiveness of LLMs for recommendations

Test our H&M LLM-powered personalized recommender system

💻 Explore all the lessons and the code in our freely available GitHub repository.


Figure 2: End-to-end architecture of an H&M real-time personalized recommender powered by LLMs
Why add LLMs to our recommender?
Traditional recommender systems face several limitations:

They struggle to capture nuanced user preferences that are better expressed in natural language

They require extensive training data and frequent retraining to adapt to new patterns

They can't explain their recommendations in a way humans can easily understand

LLMs offer a compelling solution to these challenges. They can:

Leverage semantic understanding to grasp products and user preferences

Process natural language queries to better understand user intent

Generate human-like explanations for recommendations

Adapt to new scenarios without explicit retraining

Upgrading our current architecture with LLMs
So far, our H&M recommender has relied on classical ML techniques:

Candidate generation using a two-tower neural network

Vector similarity search for retrieving relevant items

Final ranking using a CatBoost model

In this lesson, we'll explore two architectural enhancements:

LLM-based ranking: Replace our CatBoost ranker with an LLM to score and rank candidates based on context and natural language understanding. This allows us to incorporate product descriptions, user preferences, and style considerations in our ranking decisions.

Semantic search integration: Enable users to search for products using natural language queries, making the recommendation system more intuitive and accessible. Users can express their preferences directly: "Show me summer dresses similar to what I bought last month, but in lighter fabrics."

We'll implement these changes using the same production-ready architecture with Hopsworks Serverless and KServe [4], ensuring our system remains scalable and maintainable.

Table of Contents:
Building a ranking algorithm using LLMs

Deploying the ranking service using KServe

Using semantic search for personalized recommendations

Are LLMs for personalized recommendations any good?

Testing the H&M LLM-powered personalized recommender

1 - Building a ranking algorithm using LLMs
We will combine LLMs’ natural language understanding with the existing architecture’s structured data processing to rank products based on their purchase likelihood.

This approach is inspired by state-of-the-art research in the field. For example, the LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking [2] leverages user interaction history and a verbalizer-based approach for ranking recommendations without generating long texts. Another valuable resource is the Survey on Large Language Models for Recommendation [3]. It discusses how LLMs can improve recommendation quality thanks to their extensive data training and high-quality textual representations.

Building on this research, our ranking system has three main components:

The prediction function handles the ranking logic

The prompt template guides the LLM’s behavior

An output parser processes LLM's responses

As seen in Lesson 1, the ranking algorithm is used in the 4-stage recommendation architecture. At Stage 1, a course list of candidate H&M articles is returned by querying the vector index using the customer embedding.

The ranking algorithm reorders the H&M items based on their predicted purchase likelihood. This LLM-based ranker uses detailed product attributes and customer context to enhance the final ordering of recommendations, improving the initial nearest-neighbor results.

Let's examine each component in detail.


Figure 3: Understanding how the offline pipeline connects to the online pipeline
The Prediction Function

The heart of our ranking system lies in the predict() method:

class Predict(object):
    ... # Other methods

    def predict(self, inputs):
        features = inputs[0].pop("ranking_features")[:20]
        article_ids = inputs[0].pop("article_ids")[:20]
        preprocessed_features = self._preprocess_features(features)
    
        scores = []
        for candidate in preprocessed_features_candidates:
            try:
                text = self.llm.invoke(candidate)
                score = self.parser.parse(text)
            except Exception as exception:
                logging.error(exception)
                # Add minimum default score in case of error
                score = 0
            scores.append(score)
    
        return {"scores": scores, "article_ids": article_ids}
The prediction function processes product features through the LLM in batches of 20 candidates, maintaining a reasonable inference time while ensuring reliable scoring. Each product's features are preprocessed, scored by the LLM, and paired with its article ID for tracking.

The Prompt Template

The prompt template structures the interaction with the LLM.

The prompt is designed to:

Define the assistant's role

Specify the exact format for input features

Handle both numeric and categorical features

Request a precise probability output with 4-decimal precision

PROMPT_TEMPLATE = """
You are a helpful assistant specialized in predicting customer behavior. 
Your task is to analyze the features of a product and predict the 
probability of it being purchased by a customer.

### Instructions:
1. Use the provided features of the product to make your prediction.
2. Consider the following numeric and categorical features:
   - Numeric features: These are quantitative attributes
   - Categorical features: These describe qualitative aspects

### Product and User Features:
Numeric features:
- Age: {age}
- Month Sin: {month_sin}
- Month Cos: {month_cos}

Categorical features:
- Product Type: {product_type_name}
- Product Group: {product_group_name}
- Graphical Appearance: {graphical_appearance_name}
- Colour Group: {colour_group_name}
- Perceived Colour Value: {perceived_colour_value_name}
- Perceived Colour Master Value: {perceived_colour_master_name}
- Department Name: {department_name}
- Index Name: {index_name}
- Department: {index_group_name}
- Sub-Department: {section_name}
- Group: {garment_group_name}

### Your Task:
Based on the features provided, predict the probability that the 
customer will purchase this product to 4-decimals precision.

Provide the output in the following format:
Probability: """
Parsing the Output

The ScoreOutputParser class leverages Pyndatic to ensure the extracted probabilities are valid numbers between 0 and 1, with error handling for malformed responses.

class ScoreOutputParser(BaseOutputParser[float]):
    def parse(self, output) -> float:
        text = output['text']
        probability_str = text.split("Probability:")[1].strip()
        probability = float(probability_str)
        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability value must be between 0 and 1.")
        return probability
The ScoreOutputParser:

Extracts the probability value from the LLM's response

Convert the extracted value to a float

Validates the float is between 0 and 1

Returns the parsed score

Building with LangChain

Finally, we connect everything with LangChain:

class Predict(object):
   ... # Other methods

    def _build_lang_chain(self):
        model = ChatOpenAI(
            model_name='gpt-4o-mini-2024-07-18',
            temperature=0.7,
            openai_api_key=self.openai_api_key,
        )
        prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=PROMPT_TEMPLATE,
        )
        langchain = LLMChain(
            llm=model,
            prompt=prompt,
            verbose=True
        )

        return langchain
This setup:

Initializes the LLM model

Creates the prompt template

Combines them into a chain for processing

In addition, we use Pydantic for data validation and settings management.

The error handling implementation includes:

Default scores of 0 for failed predictions

Validation of probability ranges

Logging of errors for debugging

This ensures that the data entering our models is validated and correctly structured, catching errors early and maintaining robust data handling. By combining LangChain and Pydantic, we balance the LLM's natural language capabilities with the need for reliable, structured outputs.


Figure 4: Building an H&M real-time personalized recommender using the 4-stage architecture
Access the entire Python class on our GitHub.

2 - Deploying the ranking service using KServe
Our LLM-based ranking deployment uses the same infrastructure built in Lesson 4, leveraging Kubernetes and KServe through Hopsworks serverless.

The query Transformer class
The online inference pipeline logic, which in our case follows the 4-stage design for recommenders, starts with the query Transformer class.

As explained in Lesson 4, the query Transformer computes the customer query embedding (stage 1) and then calls the ranking server for the rest of the steps.


Figure 5: Deploying the online inference pipeline to Hopsworks Serverless using KServe
Thus, we implemented a flexible deployment architecture that uses a single line of code to route requests between our classical CatBoost and LLM-based rankers.

class Transformer(object):
    def __init__(self) -> None:
        ... Other attributes

        self.ranking_server = ms.get_deployment(self.ranking_model_type)
This setup allows us to dynamically switch between ranking models using the ranking_model_type environment variable, which can be set to:

"ranking" for the classical CatBoost ranker

"llmranking" for our new LLM-based ranker

The Transformer class orchestrates this process through three key stages:

Initialization: Connects to Hopsworks and sets up the appropriate ranking model based on environment configuration

Preprocessing: Extract customer information, computes temporal features, and format inputs consistently for both rankers

Postprocessing: Returns ranked predictions in a standardized format, regardless of which ranking model was used

This unified pipeline allows us to experiment with different ranking approaches while maintaining consistent feature processing and prediction serving across all deployments without requiring any changes to our infrastructure code.

Environment Configuration in KServe
Our deployment settings are managed through a dedicated secrets preparation method:

class HopsworksQueryModel:
   ... # Other methods

    @classmethod
    def _prepare_secrets(cls, ranking_model_type: Literal["ranking", "llmranking"]):
        connection = hopsworks.connection(
            host="c.app.hopsworks.ai",
            hostname_verification=False,
            port=443,
            api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value(),
        )

        secrets_api = connection.get_secrets_api()
        secrets = secrets_api.get_secrets()
        existing_secret_keys = [secret.name for secret in secrets]
        if "RANKING_MODEL_TYPE" in existing_secret_keys:
            secrets_api._delete(name="RANKING_MODEL_TYPE")

        project = hopsworks.login()
        secrets_api.create_secret(
            "RANKING_MODEL_TYPE",
            ranking_model_type,
            project=project.name,
        )
This approach is especially valuable in production environments, as it allows us to manage multiple deployment variants with different settings, all controlled via environment variables.

In practice, this means we can:

Run A/B tests with multiple configurations

Maintain separate testing and production environments

Quickly switch ranking models without redeploying the entire system

Deploying the ranking service using KServe
Since we maintain the same input/output interface across both CatBoost and LLM-based predictor, we can leverage the same Ranking Inference Pipeline Transformer class.

This ensures the preprocessing and postprocessing steps remain consistent across both ranking approaches.

During initialization, the Transformer class:

Connects to Hopsworks and sets up the feature store

Retrieves necessary feature groups and views: transactions, articles, customers, and candidate embeddings

Gets the ranking model from the model registry (either CatBoost or LLM-based)

Prepares the feature view for batch scoring

Extracts the required feature names for the ranking model

The preprocess method transforms raw inputs into the format needed for ranking:

class Transformer(object):
   ...

    def preprocess(self, inputs):
        # Extract customer ID and find nearest neighbors
        customer_id = inputs["instances"][0]["customer_id"]
        neighbors = self.candidate_index.find_neighbors(
            inputs["instances"][0]["query_emb"],
            k=100,
        )
    
        # Filter out already purchased items
        already_bought_items_ids =   self.transactions_fg.select("article_id")
            .filter(self.transactions_fg.customer_id==customer_id)
            .read(dataframe_type="pandas").values.reshape(-1).tolist()
    
         # Prepare features for ranking
         # [... feature preparation code ...]
    
        return {
          "inputs": [{
            "ranking_features": ranking_model_inputs.values.tolist(),
            "article_ids": item_id_list,
          }]
        }
The postprocess() method handles the output standardization:

class Transformer(object):
   ...

    def postprocess(self, outputs):
        ranking = list(zip(outputs["scores"], outputs["article_ids"]))
        ranking.sort(reverse=True)

        return {"ranking": ranking}
The LLM-based predictor replaces the CatBoost predictor while maintaining compatibility with the KServe interface.

class Predict(object):
    def __init__(self):
        self.input_variables = ["age", "month_sin", "month_cos", "product_type_name", 
                              "product_group_name", ...]
        self._retrieve_secrets()
        self.llm = self._build_lang_chain()
        self.parser = ScoreOutputParser()

    def predict(self, inputs):
        features = inputs[0].pop("ranking_features")[:20]
        article_ids = inputs[0].pop("article_ids")[:20]
        
        preprocessed_features = self._preprocess_features(features)
        scores = []
        
        for candidate in preprocessed_features:
            try:
                text = self.llm.invoke(candidate)
                score = self.parser.parse(text)
            except Exception as exception:
                score = 0
            scores.append(score)
            
        return {
            "scores": scores,
            "article_ids": article_ids,
        }
The predictor is wrapped in a KServe-compatible class that implements the required interface:

Inherits from KServe's Model class

Implements load() and predict() methods

Maintains compatibility with the transformer pipeline

This modular setup makes deployment easier and allows for future experiments with new ranking methods without needing major infrastructure changes.

By keeping this consistent interface, we can easily switch between the CatBoost and LLM-based ranking approaches while reusing the same preprocessing and postprocessing pipeline.

LLM ranking deployment
To deploy the LLM ranker, we will create a new deployment that will override the classic one. This new deployment will include both the query and ranking inference pipelines.

This deployment leverages the Transformer class and the ranking_model_type environment variable to dynamically route requests to either the classical CatBoost ranker or the LLM-based ranker.

The deployment of the LLM ranker involves:

Model registration

Deployment

Starting the service

Testing functionality using the provided inference pipeline

Retrieve Top Recommendations

Debugging

1 - Register the LLM ranking model

First, register the LLM ranking model in the Hopsworks Model Registry to ensure it is properly versioned and ready for deployment:

ranking_model = hopsworks_integration.llm_ranking_serving.HopsworksLLMRankingModel()
ranking_model.register(project.get_model_registry())
After registration, you can explore the model in Hopsworks Serverless: Data Science → Model Registry.

2 - Deploy the LLM ranking model

Initiate the deployment process using the deploy() method of the HopsworksLLMRankingModel class. This sets up the necessary infrastructure to serve the LLM ranker as a real-time service:

ranking_deployment = hopsworks_integration.llm_ranking_serving.HopsworksLLMRankingModel.deploy()
3 - Start the deployment

Once the deployment is created, start it explicitly using the start() method to activate the ranking service:

ranking_deployment.start()
Explore the deployments in Hopsworks Serverless: Data Science → Deployments.

4 - Test the deployment

Create a test example to verify the ranking deployment. Use the predict() method to perform inference on the deployed LLM ranker with features such as customer_id, query_emb (query embeddings), and temporal features (month_sin, month_cos):

test_ranking_input = [
    {
        "customer_id": "d327d0ad9e30085a436933d...",
        "month_sin": 1.2246467991473532e-16,
        "query_emb": [
            0.214135289, 0.571055949, 0.330709577, -0.225899458,
            ...
        ],
        "month_cos": -1.0,
    }
]

ranked_candidates = ranking_deployment.predict(inputs=test_ranking_input)
5 - Retrieve top recommendations

Use the get_top_recommendations() function to extract the top-ranked article IDs based on the predictions returned by the LLM ranker:

def get_top_recommendations(ranked_candidates, k=3):
    return [candidate[-1] for candidate in ranked_candidates["ranking"][:k]]

recommendations = get_top_recommendations(ranked_candidates["predictions"], k=3)
6 - Debugging logs

In case of failures, access the logs through Kibana for debugging. The logs provide detailed information about the inference pipeline and any errors that may have occurred:

ranking_deployment.get_logs(component="predictor", tail=200)
🔗 Access full code

3 - Using semantic search for personalized recommendations
Now that we understand how our system leverages semantic search with LLMs, let's explore how it integrates into a broader framework for personalized product recommendations.

The system operates with a structure similar to RAG but is adapted for product recommendations:

1. Embedding & indexing phase

Embedding: Process product descriptions and metadata through an embedding model

Indexing: Store these embeddings in a vector index, which acts as our "knowledge base" of products maintained in Hopsworks

2. Retrieval phase

When a user interacts with the system:

Query generation: Generate a query based on the users’ input and profile information

Retrieval: Retrieve similar products from the vector index using the query

Result: Return the most relevant products based on vector similarity

3. Personalization with LLM

In this system, there’s no generation or ranking phase. Instead, we:

Use an LLM to map the user profile and input query to a list of generated product candidates.

Use these LLM-generated items in the semantic search process to access similar H&M fashion articles in stock.

Combines embeddings with user preferences and product metadata for personalized recommendations

Figure 6 shows a high-level overview of the system operations.


Figure 6: Semantic search for H&M personalized recommendations
The recommendation pipeline
Our recommendation pipeline consists of several key components:

1 - Feature Pipeline

Input: Takes H&M article data as input

Processing: Cleans H&M product descriptions and converts them into vectors using an embedding model

Storage: Stores article embeddings along with metadata

2 - Vector Index Layer

Maintenance: Stored in Hopsworks

Storage: Contains both article embeddings and metadata

Function: Enables efficient similarity search across the product catalog and retrieves similar articles based on the H&M articles description

3 - Personalization with LLM

Input: Takes user input and profile information

Processing: An LLM generates 3 to 5 fashion articles leveraging user profiles and preferences. As the LLM-generated items do not exist, we will use them as a proxy to query the H&M fashion articles index, returning items similar to those generated by the LLM.

Vector index for semantic search on fashion item descriptions
In Lesson 4, we used a pre-trained Sentence Transformer model all-MiniLM-L6-v2 to create embeddings for article descriptions.

These embeddings map each article description in a vector space, enabling semantic similarities comparison between articles.

Below is the code snippet used for creating the embeddings:

for i, desc in enumerate(articles_df["article_description"].head(n=3)):
    logger.info(f"Item {i+1}:\n{desc}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(
    f"Loading '{settings.FEATURES_EMBEDDING_MODEL_ID}' embedding model to {device=}"
)

model = SentenceTransformer(settings.FEATURES_EMBEDDING_MODEL_ID, device=device)
articles_df = generate_embeddings_for_dataframe(
    articles_df, "article_description", model, batch_size=128
)  # Adjust batch size as needed.
To retrieve articles based on semantic similarity, the embeddings must be accessible through a feature view. This view combines multiple feature groups, ensuring the retrieval model has all the necessary features for downstream tasks, all achieved using the create_articles_feature_group function.

def create_articles_feature_group(
        fs,
        df: pd.DataFrame,
        articles_description_embedding_dim: int,
        online_enabled: bool = True,
):
    emb = embedding.EmbeddingIndex()
    emb.add_embedding("embeddings", articles_description_embedding_dim)

    articles_fg = fs.get_or_create_feature_group(
        name="articles",
        version=1,
        description="Fashion items data including type of item, visual description and category",
        primary_key=["article_id"],
        online_enabled=online_enabled,
        features=constants.article_feature_description,
        embedding_index=emb,
    )
    articles_fg.insert(df, wait=True)

    return articles_fg
The EmbeddingIndex object was initialized using the following snippet:

emb = embedding.EmbeddingIndex()
emb.add_embedding("embeddings", articles_description_embedding_dim)
This setup defined the embeddings column as the vector index, specifying its dimensionality through articles_description_embedding_dim. This configuration enables efficient similarity searches.

By joining this feature group with others in a feature view, we ensured all necessary data was accessible for retrieval models:

def create_retrieval_feature_view(fs):
    trans_fg = fs.get_feature_group(name="transactions", version=1)
    customers_fg = fs.get_feature_group(name="customers", version=1)
    articles_fg = fs.get_feature_group(name="articles", version=1)

    selected_features = (
        trans_fg.select(
            ["customer_id", "article_id", "t_dat", "price", "month_sin", "month_cos"]
        )
        .join(
            customers_fg.select(["age", "club_member_status", "age_group"]),
            on="customer_id",
        )
        .join(
            articles_fg.select(["garment_group_name", "index_group_name"]),
            on="article_id",
        )
    )

    feature_view = fs.get_or_create_feature_view(
        name="retrieval",
        query=selected_features,
        version=1,
    )

    return feature_view
Dynamic recommendations with LLMs
The llm_recommendations() function is the backbone of our personalized recommendation system, combining LLM-based reasoning with semantic search and state management to provide tailored suggestions for users.

Below, we break down its key steps and their integration into the broader recommendation pipeline.

1 - Initializing the recommendation system

The function begins by setting up the necessary environment for generating and displaying recommendations.

This includes initializing the session state to track user interactions, loading the embedding model, and configuring the interface for user input.

# Initialize session state and load embedding model
initialize_llm_state()
embedding_model = SentenceTransformer(settings.FEATURES_EMBEDDING_MODEL_ID)

# Gender selection
gender = st.selectbox("Select gender:", ("Male", "Female"))

# Input options for fashion needs
input_options = [
    "I'm going to the beach for a week-long vacation. What items do I need?",
    "I have a formal winter wedding to attend next month. What should I wear?",
    "I'm starting a new job at a tech startup with a casual dress code. What items should I add to my wardrobe?",
    "Custom input",
]
selected_input = st.selectbox("Choose your fashion need or enter a custom one:", input_options)

# Handle custom user input if selected
if selected_input == "Custom input":
    user_request = st.text_input("Enter your custom fashion need:")
else:
    user_request = selected_input
This setup captures user preferences, including gender and specific fashion needs, which are later used to tailor the recommendations.

2 - Generating initial recommendations

When the user clicks the “Get LLM Recommendations” button, the system activates the LLM to generate context-aware recommendations. These are tailored to the user’s input and gender.

if st.button("Get LLM Recommendations") and user_request:
    with st.spinner("Generating recommendations..."):
        fashion_chain = get_fashion_chain(api_key)
        item_recommendations, summary = get_fashion_recommendations(
            user_request, fashion_chain, gender
        )
The output includes a categorized list of recommended items and a summary of the outfit or style suggested.

3 - Integrating semantic search for similar items

The system uses semantic search to retrieve products similar to the LLM-generated descriptions, leveraging embeddings stored in the vector index.

similar_items = get_similar_items(description, embedding_model, articles_fv)
4 - Real-time user interaction and updates

The system dynamically tracks user interactions, such as purchases, and updates the recommendations in real time. For example, if a user marks an item as purchased, a replacement is automatically selected from the extra items.

# Track purchased items and update recommendations
tracker.track_shown_items(customer_id, [(item[1][0], 0.0) for item in shown_items])

if was_purchased:
    category_updated = True
    extra_items = st.session_state.llm_extra_items.get(category, [])
    if extra_items:
        new_item = extra_items.pop(0)
        remaining_items.append(new_item)
This mechanism ensures that the recommendations evolve based on user feedback, maintaining a dynamic and personalized experience.

5 - Displaying recommendations and summaries

The function organizes recommendations into categories, each accompanied by visual and textual details. It also provides a summary of the outfit or style, enhancing the user experience.

# Display outfit summary if available
if st.session_state.outfit_summary:
    st.markdown("## 🎨 Outfit Summary")
    st.markdown(
        f"<h3 style='font-size: 20px;'>{st.session_state.outfit_summary}</h3>",
        unsafe_allow_html=True,
    )
The recommendations are displayed in a clean, organized format, allowing users to browse and interact with the items easily.

Decomposing key functions in the LLM recommender
In the previous section, we described the high-level flow of how the system generates personalized fashion recommendations. Now, let’s dive deeper into the core functions that drive this process, ensuring that each step works in harmony to deliver tailored, context-aware suggestions.

The get_fashion_chain() function and its prompt template

The first critical piece is the get_fashion_chain() function, which sets up the conversational model used for generating recommendations.

This function uses a prompt template that structures the LLM’s response by defining how it should categorize, describe, and summarize the recommended items.

The prompt is designed to generate 3-5 fashion items that follow context-specific recommendations, providing a clear format for the LLM’s outputs.

template = """
    You are a fashion recommender for H&M. 
    
    Customer request: {user_input}
    
    Gender: {gender}
    
    Generate 3-5 necessary fashion items with detailed descriptions, tailored for an H&M-style dataset and appropriate for the specified gender. 
    Each item description should be specific, suitable for creating embeddings, and relevant to the gender. 

    ...

    Example for male gender:
    👖 Pants @ Slim-fit dark wash jeans with subtle distressing | 👕 Top @ Classic 

    ...
"""
The get_fashion_recommendations() function

Once the fashion chain is initialized, the get_fashion_recommendations() function takes over.

It receives user input and gender preferences and passes them to the LLM chain to produce a set of recommendations.

This function carefully parses the LLM’s response into categorized suggestions and an overall outfit summary that aligns with the user’s needs and preferences.

def get_fashion_recommendations(user_input, fashion_chain, gender):
    response = fashion_chain.run(user_input=user_input, gender=gender)
    items = response.strip().split(" | ")

    outfit_summary = items[-1] if len(items) > 1 else "No summary available."
    item_descriptions = items[:-1] if len(items) > 1 else items

    parsed_items = []
    for item in item_descriptions:
        try:
            emoji_category, description = item.split(" @ ", 1)
            emoji, category = emoji_category.split(" ", 1)
            parsed_items.append((emoji, category, description))
        except ValueError:
            parsed_items.append(("🔷", "Item", item))

    return parsed_items, outfit_summary
The get_similar_items() function

Now that we have a set of fashion items generated by the LLM, we must match them with our actual H&M inventory before showing them to the user.

The LLM-generated items are a proxy between the user’s preferences and the real H&M inventory.

We cannot directly use semantic search between the user query and item description because the semantics of the two are different.

But by mapping the user query (plus the customer features) to a set of article descriptions, we can successfully perform a semantic search between the generated and the real H&M fashion article descriptions:

def get_similar_items(description, embedding_model, articles_fv):
    description_embedding = embedding_model.encode(description)

    return articles_fv.find_neighbors(description_embedding, k=25)
Streamlit code for content display

Finally, the entire recommendation process is presented to the user through a Streamlit interface.

This section of the code handles the visualization of recommendations, displaying categories, individual items, and outfit summaries.

It also integrates interactive features that allow users to engage with the recommendations and track their actions, such as making a purchase or revisiting previous suggestions.

The entire semantic search for recommendations code is available on our GitHub.

4 - Are LLMs for Personalized Recommendations Any Good?
Using LLMs for personalized recommendations is an exciting approach that opens up many opportunities, but it comes with challenges that make it an experimental exploration at this stage.

Early and experimental: While research and early experiments are promising, this approach is still in its infancy. The integration of LLMs into large-scale recommendation systems is a developing field, and established best practices have yet to be fully defined.

Cost implications: One of the biggest challenges with LLMs is their computational cost. Running LLMs in real-time for ranking or semantic search requires substantial hardware resources. This can significantly increase operational expenses compared to traditional ML methods.

Latency and scalability: LLMs can introduce higher latency, which may negatively impact user experience, especially in real-time systems. Scaling these systems to serve millions of users simultaneously requires careful planning, optimization, and substantial infrastructure investments.

Potential for bias and overfitting: LLMs heavily rely on their training data, which can lead to biases or an overemphasis on textual features. This could result in suboptimal recommendations, especially if structured data signals like customer purchase history or behavioral patterns are not properly integrated.

Research and innovation needed: Further research is essential to understand how LLMs can complement or enhance classical recommender system architectures. Questions regarding hybrid approaches, efficient fine-tuning, and the trade-offs between accuracy, latency, and cost require deeper exploration before LLMs can be deemed production-ready.

In conclusion, while LLMs offer intriguing possibilities for personalized recommendations, they are not yet a silver bullet.

For now, they are best suited for experimental setups, PoCs, niche applications, or as a complementary layer to existing recommender systems.

5 - Testing the H&M LLM-Powered Personalized Recommender
We are finally here: the moment where we can test our H&M real-time personalized recommender powered by LLMs!

To make testing seamless, we’ve built a simple Streamlit app that allows us to visualize real-time recommendations for different users and interact with the system.

The app also lets you generate new user interactions to see how recommendations adapt over time. While we won’t dive into the Streamlit code itself, under the hood, calling the KServe LLM ranking deployment through Hopsworks is as simple as:

ranked_candidates = ranking_deployment.predict(inputs=test_ranking_input)
Everything else is Streamlit code, which you can find in our GitHub repository.

Deploy the KServe LLM Ranking System
First, deploy the LLM ranker using the provided make command:

make create-deployments-llm-ranking
After running this command, you can view and manage the deployment in Hopsworks Serverless → Data Science → Deployments.

Now, we can use our H&M personalized recommender using the LLM ranking service instead of the CatBoost one.

For the deployments to run successfully, you must first run the feature and training pipelines: do that with a few commands you can find on our GitHub ⇠

Use the Streamlit App with "LLM Recommendations"
Also, our Streamlit app has been updated to include an “LLM Recommendations” feature to test out the semantic search recommender.

This allows us to interactively test the semantic search feature, explore recommendations for different users, and compare the results to the standard recommender.

You can start the app by running:

make start-ui-llm-ranking
Once started, the app will be accessible at http://localhost:8501/

Find a step-by-step installation and usage guide on our GitHub ⇠

Conclusion
Congratulations on completing the "Hands-On Real-time Personalized Recommender" course! You've come a long way– from building basic recommendation systems to implementing sophisticated LLM-powered recommenders.

Throughout these five lessons, you've learned to:

Build and deploy a production-ready H&M real-time personalized recommender

Implement feature and training pipelines

Create scalable deployment architectures

Enhance recommendations with LLM capabilities

Whether you're working on e-commerce platforms like H&M or content recommendation systems like TikTok, the principles and architectures covered in this course provide a solid foundation for your recommender system projects.

Thank you for learning with Decoding ML and Hopsworks. We hope this course helps you create fantastic recommendation systems for your future projects!

💻 Explore all the lessons and the code in our freely available GitHub repository.

If you have questions or need clarification, feel free to ask. See you in the next session!

References
Literature
[1] Decodingml. (n.d.). GitHub - decodingml/personalized-recommender-course. GitHub. https://github.com/decodingml/personalized-recommender-course

[2] Yue, Z., Rabhi, S., Moreira, G. de S. P., Wang, D., & Oldridge, E. (2023). LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.

[3] Wu, L., Zheng, Z., Qiu, Z., Wang, H., Gu, H., Shen, T., Qin, C., Zhu, C., Zhu, H., Liu, Q., Xiong, H., & Chen, E. (2023). A Survey on Large Language Models for Recommendation.

[4] Hopsworks. (n.d.). What is Kserve? - Hopsworks. https://www.hopsworks.ai/dictionary/kserve

Images
If not otherwise stated, all images are created by the author.

Sponsors
Thank our sponsors for supporting our work!


Subscribe to Decoding ML
Launched 2 years ago
Join for proven content on designing, coding, and deploying production-grade AI systems with software engineering and MLOps best practices to help you ship AI applications. Every week, straight to your inbox.
Type your email...
Subscribe
By subscribing, I agree to Substack's Terms of Use, and acknowledge its Information Collection Notice and Privacy Policy.
28 Likes
∙
12 Restacks
Discussion about this post
Write a comment...
Cha_le
Feb 19

Thank you so much for this.

I have learn a lot from this.

Like (1)
Reply
Share

Build your Second Brain AI assistant
Using agents, RAG, LLMOps and LLM systems
Feb 6 • Paul Iusztin
870
35

LLMOps for production agentic RAG
Evaluating and monitoring LLM agents with SmolAgents and Opik
Mar 20 • Paul Iusztin and Anca Ioana Muscalagiu
94

Playbook to fine-tune and deploy LLMs
Specialized open-source LLMs for production
Mar 6 • Paul Iusztin
90
4

Ready for more?

Type your email...
Subscribe
© 2025 Paul Iusztin
Privacy ∙ Terms ∙ Collection notice
Start writing
Get the app
Substack is the home for great culture

you lossed the chats but we had agreed and come up with the plan below:
 
Plan:

**I. Core Architecture (Remains the Same Fundamentally)**

*   **Two-Tower Model:** User Tower & Post Tower.
*   **4-Stage Recommender System (Online Inference):** Candidate Generation, Filtering, Ranking, Re-ranking.
*   **FTI (Feature/Training/Inference) Pipelines.**
*   **Technology Stack:** Open-source (Milvus, Feast, MLflow, Airflow/Kubeflow, KServe) and GCP (Firestore, GCS, BigQuery, GKE, Vertex AI for potential components), ScyllaDB.

**II. Detailed Plan & Implementation Phases (Revised)**

**Phase 0: Setup & Foundational Data Pipelines**

1.  **Data Ingestion & Storage:**
    *   **Posts:** Firestore (source of truth).
    *   **User Profiles:** Firestore or accessible DB.
    *   **Interaction Logs & Impressions (Your Existing Real-time Flow):**
        *   **Cloud Functions** trigger on DB events (new post, like, etc.).
        *   Signals are sent to your **FastAPI app running in a VM**.
        *   **Actions for your FastAPI App:**
            *   **Real-time "Seen" Post Updates:** The FastAPI app must update **ScyllaDB** with `(user_id, post_id)` information for posts that have been viewed/impressed upon. This will be used by the filtering stage.
            *   **Batch Data Accumulation:** The FastAPI app needs to ensure interaction data is also reliably written to **GCS** (e.g., as JSONL files in hourly/daily batches). This GCS data will then be loaded into **BigQuery** for batch feature computation by Feast. *Consider if your FastAPI app can directly write to a Pub/Sub topic which then reliably feeds GCS/BigQuery, as this provides good decoupling and resilience for the batch path.*
    *   **User-Selected Categories (Onboarding):** Stored with user profile, accessible.
    *   **User's Preferred/Interacted Categories:** This will be derived from onboarding and interaction history.

**Phase 1: Feature Engineering**

*   **Tool:** **Feast** (self-hosted or on GKE, using BigQuery for offline store, potentially Redis/Cloud SQL for some online user features if low latency is critical beyond what batch provides).
*   **User Features:**
    *   `uid` (ID).
    *   `about`, `headline`: Text embeddings (Sentence Transformer), extracted keywords.
    *   `selected_categories_onboarding`: List of category IDs (one-hot encoded or embedded).
    *   Aggregated historical interactions (from BigQuery via Feast):
        *   Counts of likes, comments, shares, bookmarks, `VIEW_FULL`, dislikes, reports (over various time windows).
        *   Preferred `mediaType` (based on interaction counts).
        *   Interaction counts per `category_id` (e.g., `likes_in_category_X`, `views_in_category_Y`).
        *   Affinity scores for each `category_id` based on positive interactions.
        *   Average embedding of positively interacted posts.
*   **Post Features:**
    *   `post_id` (ID).
    *   `description`: Text embedding (Sentence Transformer) – primary for Post Tower & Milvus.
    *   `category_id`: Categorical feature (one-hot encoded or entity embedded).
    *   `mediaType`: Categorical feature (one-hot encoded or entity embedded).
    *   `creator_id`: Categorical feature.
    *   `createdAt`: Timestamp (for `post_age_hours`, `days_since_creation`).
    *   Engagement features (normalized): `viewCount`, `upvoteCount`, `downvoteCount`, `commentsCount`, `shareCount`, `bookmarkCount`.
*   **User-Post Interaction Features (for Ranking Model):**
    *   Is the post's `category_id` one of the user's `selected_categories_onboarding`? (Boolean)
    *   Is the post's `category_id` among the user's top N interacted categories?
    *   Has the user interacted with other posts from this `creator_id`?
    *   Cosine similarity between user's average interacted post embedding and current post's description embedding.
    *   User's historical interaction rate (e.g., likes/impression) within the post's `category_id`.

**Phase 2: Model Training (Two-Tower & Ranking)**

*   **Orchestration:** Apache Airflow (Cloud Composer) or Kubeflow Pipelines (Vertex AI Pipelines).
*   **Model Registry:** MLflow Model Registry.

1.  **Two-Tower Model Training:**
    *   **User Tower Input Features:** `uid`, embeddings of `selected_categories_onboarding`, aggregated category interaction features, profile text embeddings.
    *   **Post Tower Input Features:** `post_id`, `description` embedding, `category_id` embedding, `mediaType` embedding, `creator_id` embedding.
    *   **Labels:**
        *   Positive pairs: (`user_id`, `post_id`) from strong positive interactions.
        *   Negative pairs: (`user_id`, `post_id`) from impressions with no positive interaction, or randomly sampled non-interacted posts (potentially biased towards user's preferred categories to make negatives more challenging).
    *   **Framework:** TensorFlow/Keras or PyTorch.
    *   **Output:** Trained User Tower & Post Tower models (store in MLflow).

2.  **Ranking Model Training:**
    *   **Objective:** Predict probability of positive interaction.
    *   **Model Options:** XGBoost, LightGBM, Neural Network.
    *   **Input Features:** Relevant user, post, and user-post interaction features from Phase 1.
    *   **Labels:** Positive (1) for strong positive interactions, Negative (0) for impressions without positive interaction.
    *   **Output:** Trained ranking model (store in MLflow).

**Phase 3: Offline Inference & Candidate Indexing**

1.  **Post Embedding Generation:**
    *   **Batch Pipeline (Airflow/Kubeflow):**
        *   Loads Post Tower model from MLflow.
        *   Fetches posts from Firestore.
        *   Generates embeddings.
        *   Writes (`post_id`, `embedding`, `category_id`) to **Milvus**.
    *   **Near Real-Time for New Posts (via your existing FastAPI service):**
        *   When Cloud Function notifies FastAPI of a new post:
            *   FastAPI app loads/accesses the Post Tower model (could be a separate KServe endpoint or loaded in the VM if resources allow).
            *   Generates embedding.
            *   Inserts (`post_id`, `embedding`, `category_id`) into Milvus. This ensures the <3 min reflection time.

**Phase 4: Online Inference Deployment (4-Stage System)**

*   **Deployment Platform:** KServe on GKE.

1.  **Stage 1: Candidate Generation Service (KServe Predictor + Transformer)**
    *   **Input:** `user_id`, list of user's preferred `category_id`s (from onboarding/derived preferences).
    *   **User Embedding Generation (Transformer/Predictor):**
        *   Transformer fetches user features from Feast.
        *   Predictor (User Tower) computes user embedding.
    *   **ANN Search (Transformer):**
        *   Query Milvus with user embedding.
        *   **Filter in Milvus:** By the user's preferred/interacted `category_id`s. This is key to keeping candidates relevant to user interests.
    *   **Output:** List of candidate `post_id`s.

2.  **Stage 2: Filtering Service**
    *   **Input:** Candidate `post_id`s, `user_id`.
    *   **Process:**
        *   Check against "seen" posts for the user using **ScyllaDB**. This could be a Bloom Filter stored/updated in ScyllaDB, or direct key lookups if `(user_id, post_id)` entries are stored.
        *   Remove seen posts.
        *   Remove posts disliked/reported.
    *   **Output:** Filtered candidate `post_id`s.

3.  **Stage 3: Ranking Service (KServe Predictor + Transformer)**
    *   **Input:** Filtered `post_id`s, `user_id`.
    *   **Feature Enrichment (Transformer):** Fetch full post features (Firestore/Feast), user features (Feast). Construct ranking model input.
    *   **Scoring (Predictor):** Load Ranking Model from MLflow, predict scores.
    *   **Output:** (`post_id`, `score`) tuples.

4.  **Stage 4: Re-ranking & Presentation Service**
    *   **Input:** (`post_id`, `score`) tuples.
    *   **Process:**
        *   Order by score.
        *   **Category Balancing/Diversity:** Ensure not all posts are from a single category if the user has multiple interests. Implement logic to interleave posts from different relevant categories.
        *   New post boosts.
    *   **Output:** Final ordered list of `post_id`s.

**Phase 5: Handling Cold Starts**

1.  **New User Cold Start:**
    *   Use `selected_categories_onboarding` to:
        *   Form an initial user embedding (average embedding of those categories or users interested in them).
        *   Initially recommend popular/recent posts from these selected categories.
2.  **New Post Cold Start:** Boost in Stage 4, content embedding allows discovery.

**Phase 6: Monitoring, Evaluation & Iteration (As before)**

*   Focus on metrics related to engagement within preferred categories.

**Summary of Key Changes Based on Your Feedback:**

*   **"Spaces" are out, "Categories" are in:** All logic now revolves around post categories and user category preferences.
*   **Real-time Ingestion via Your FastAPI:** Your existing Cloud Function -> FastAPI flow is the entry point for real-time signals. This FastAPI app now has the added responsibility of updating ScyllaDB (for seen posts) and ensuring data lands in GCS/BigQuery for batch processing.
*   **ScyllaDB for Seen Posts:** Filtering stage will query ScyllaDB.
*   **Category Balancing:** Added to the re-ranking stage.

**Remaining Questions/Considerations:**

1.  **FastAPI to GCS/BigQuery Path:**
    *   How will the FastAPI app reliably write interaction data for batch processing?
        *   Option A: FastAPI writes to local files, and a cron job on the VM uploads these to GCS daily/hourly. (Less robust).
        *   Option B: FastAPI app directly writes to GCS buckets. (Better).
        *   Option C: FastAPI app sends a message to a new Pub/Sub topic, which then uses a Dataflow job or Pub/Sub subscription to write to GCS/BigQuery. (Most robust and scalable for the batch path, even if Pub/Sub isn't used for the *initial* Cloud Function -> FastAPI hop). This adds a component but ensures the batch pipeline is well-fed.
2.  **ScyllaDB for "Seen" Posts Implementation:**
    *   Are you storing individual `(user_id, post_id)` records in ScyllaDB? Or are you planning to implement a Bloom Filter on top of ScyllaDB (e.g., by storing the filter's bit array in ScyllaDB)? Individual records are simpler but could grow very large. Bloom filters are more complex to manage but very space-efficient for "exists" checks.
3.  **Post Tower Model Access for FastAPI (New Posts):** How will the FastAPI app access the Post Tower model to embed new posts in near real-time?
    *   Option A: Load the model directly in the FastAPI VM (consumes VM resources).
    *   Option B: Deploy the Post Tower as a separate, lightweight KServe endpoint that the FastAPI app can call. (Cleaner, more scalable).
 
 