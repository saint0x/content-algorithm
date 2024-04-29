import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

class ContentRecommendationSystem:
    def __init__(self):
        self.user_profiles = {}  # Store user profiles with interests
        self.content_pool = {}   # Store content with associated interests
        self.content_scores = {}  # Store engagement scores for each content
        self.user_feedback = {}  # Store user feedback on content

    def update_user_profile(self, user_id, interests):
        self.user_profiles[user_id] = {'interests': interests, 'engagement': {}}  # Include user engagement data

    def add_content(self, content_id, text, timestamp):
        self.content_pool[content_id] = {'text': text, 'interests': [], 'engagement': 0, 'timestamp': timestamp}  # Include content engagement score and timestamp
        self.content_scores[content_id] = {'relevance': 0, 'engagement': 0, 'sentiment': 0, 'trend_score': 0}  # Initialize scores
        self.analyze_content(content_id)

    def record_user_engagement(self, user_id, content_id, engagement):
        if user_id in self.user_profiles and content_id in self.content_pool:
            self.user_profiles[user_id]['engagement'][content_id] = engagement
            self.content_pool[content_id]['engagement'] += engagement

    def recommend_content(self, user_id):
        user_interests = self.user_profiles.get(user_id, {}).get('interests', [])
        relevant_content = []

        # Calculate User Interest Scores (UIS) and User Engagement Scores (UES)
        user_interest_scores = {interest: 1 for interest in user_interests}
        user_engagement_scores = {content_id: sum(engagement.values()) for content_id, engagement in self.user_profiles.get(user_id, {}).get('engagement', {}).items()}

        # Find content relevant to user interests and calculate Content Relevance Scores (CRS)
        for content_id, data in self.content_pool.items():
            content_relevance_score = sum(user_interest_scores.get(interest, 0) for interest in data['interests'])
            self.content_scores[content_id]['relevance'] = content_relevance_score
            if content_relevance_score > 0:
                relevant_content.append(content_id)

        if not relevant_content:
            # If no relevant content found, recommend random content
            return random.choices(list(self.content_pool.keys()), k=3)

        # Apply Engagement Score (ES) boosting and calculate Overall Content Score (OCS)
        relevant_content = self.boost_content(relevant_content)
        ocs_scores = [self.calculate_ocs(content_id) for content_id in relevant_content]

        # Update content trend scores based on temporal considerations
        self.update_trend_scores()

        # Select content based on Overall Content Score (OCS) and Trend Scores
        selected_content = self.select_content(ocs_scores)

        return selected_content

    def boost_content(self, content_list):
        # Boost content dissemination based on engagement scores
        boosted_content = []
        for content_id in content_list:
            # Boost the engagement score
            self.content_scores[content_id]['engagement'] += self.content_pool[content_id]['engagement']
            boosted_content.extend([content_id] * self.content_scores[content_id]['engagement'])  # Add content multiple times based on engagement score
        return boosted_content

    def calculate_ocs(self, content_id):
        # Calculate Overall Content Score (OCS) based on relevance, engagement, and sentiment scores
        relevance_score = self.content_scores[content_id]['relevance']
        engagement_score = self.content_scores[content_id]['engagement']
        sentiment_score = self.content_scores[content_id]['sentiment']
        # Adjust weights based on importance
        weight_relevance = 0.5
        weight_engagement = 0.3
        weight_sentiment = 0.2
        ocs_score = (weight_relevance * relevance_score) + (weight_engagement * engagement_score) + (weight_sentiment * sentiment_score)
        return ocs_score

    def analyze_content(self, content_id):
        content_text = self.content_pool[content_id]['text']

        # Tokenize content text
        tokens = word_tokenize(content_text.lower())

        # Remove stopwords and perform lemmatization
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

        # Update content interests based on filtered tokens
        self.content_pool[content_id]['interests'] = filtered_tokens

        # Analyze sentiment of content text
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(content_text)['compound']

        # Update content sentiment score
        self.content_scores[content_id]['sentiment'] = sentiment_score

    def update_trend_scores(self):
        # Calculate trend scores based on temporal considerations (e.g., timestamp)
        # Example: Assign higher trend scores to more recent content
        current_time = 1000  # Placeholder for current time (can be actual timestamp)
        for content_id, data in self.content_pool.items():
            time_difference = current_time - data['timestamp']
            trend_score = 1 / (1 + time_difference)  # Inverse relationship with time difference
            self.content_scores[content_id]['trend_score'] = trend_score

    def select_content(self, ocs_scores):
        # Select content based on Overall Content Score (OCS) and Trend Scores
        # Example: Combine OCS and Trend Scores to prioritize relevant and trending content
        combined_scores = [ocs_score * self.content_scores[content_id]['trend_score'] for content_id, ocs_score in zip(self.content_pool.keys(), ocs_scores)]
        selected_content = random.choices(list(self.content_pool.keys()), k=3, weights=combined_scores)
        return selected_content

# Example usage:
recommendation_system = ContentRecommendationSystem()

# Add some user profiles
recommendation_system.update_user_profile("user1", ["technology", "cooking"])
recommendation_system.update_user_profile("user2", ["travel", "photography"])
recommendation_system.update_user_profile("user3", ["fitness", "nutrition"])

# Add some content with text and timestamps
recommendation_system.add_content("post1", "Discover the latest advancements in artificial intelligence.", 900)
recommendation_system.add_content("post2", "Learn delicious new recipes for your next cooking adventure.", 950)
recommendation_system.add_content("post3", "Explore breathtaking travel destinations around the world.", 800)
recommendation_system.add_content("post4", "Unlock the secrets to achieving your fitness goals with expert workouts.", 850)

# Simulate user engagement
recommendation_system.record_user_engagement("user1", "post1", {'likes': 10, 'comments': 5})
recommendation_system.record_user_engagement("user1", "post2", {'likes': 5, 'comments': 3})
recommendation_system.record_user_engagement("user2", "post3", {'likes': 20, 'comments': 8})
recommendation_system.record_user_engagement("user3", "post4", {'likes': 15, 'comments': 6})

# Recommend content for user1
recommended_content = recommendation_system.recommend_content("user1")
print("Recommended Content for user1:", recommended_content)
