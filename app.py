# 1. Import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 
reviews = [
    # Positive reviews
    "I loved this movie, it was fantastic and thrilling!",
    "An excellent film with superb acting.",
    "A wonderful experience, I will watch it again.",
    "The movie was good and enjoyable.",
    "Really good story and nice characters.",
    "Good acting and great direction.",
    "Amazing visuals and a gripping plot.",
    "A masterpiece with brilliant performances.",
    "Heartwarming and beautifully shot.",
    "An outstanding movie that touched my heart.",
    "I highly recommend this movie.",
    "Very entertaining and well-paced.",
    "Loved the soundtrack and the story.",
    "A delightful film with a strong message.",
    "This movie exceeded my expectations.",
    
    # Negative reviews
    "What a waste of time, the plot was terrible.",
    "The movie was boring and too long.",
    "I did not like the movie, it was dull.",
    "This movie was bad and boring.",
    "Bad plot and poor acting.",
    "Not a good movie, it was bad.",
    "Awful film, I want my money back.",
    "The story made no sense at all.",
    "Disappointing and poorly executed.",
    "Terrible acting and horrible script.",
    "I was very disappointed with this movie.",
    "The movie was a huge letdown.",
    "Slow, uninteresting, and predictable.",
    "The worst movie I have seen this year.",
    "It was painful to watch this movie."
]

labels = [
    # Positives
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # Negatives
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]



# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.33, random_state=42)

# 4. Convert text to numbers (bag of words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 7. Test with your own review
user_review = input("Write your movie review: ")
user_review_vec = vectorizer.transform([user_review])
prediction = model.predict(user_review_vec)[0]

if prediction == 1:
    print("😊 This review is Positive!")
else:
    print("😞 This review is Negative!")

