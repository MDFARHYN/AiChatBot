
# 🤖 AI Chat Bot by Farhan Ahmed

**An Open-Source, Free-to-Use AI Chat Bot for Business Solutions**

Developed to provide seamless support for businesses, this AI Chat Bot is designed to help manage customer queries effectively using advanced natural language processing. With an easy setup and customizable model, this chatbot offers relevant answers and ensures an enhanced customer experience.

## 🌟 Features
- **Open Source and Free**: Fully free to use, modify, and deploy.
- **Easy to Use**: Just upload relevant questions and answers in a CSV file, and the chatbot takes care of the rest.
- **Customizable**: You can further fine-tune the model with your own data for better accuracy.
- **Offensive Content Detection**: Built-in filters prevent the chatbot from responding to inappropriate questions.

## 🚀 How It Works
1. **Add Questions and Answers**: Simply create a CSV file with two columns: `Question` and `Answer`. Add all relevant customer queries along with their answers.
2. **Embeddings for Relevance**: The chatbot uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings, helping it identify the most relevant answer based on the user's question.
3. **Content Filtering**: To prevent the bot from responding to offensive or illegal queries, we use:
   - `toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")`
   - `zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")`
   These models detect toxic or inappropriate content and ensure that your business maintains professional interactions with users.

## 🛠️ Setup and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-chat-bot
   cd ai-chat-bot
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Your Data**:
   - Add your questions and answers in `data/questions_answers.csv`.

4. **Run the Chat Bot**:
   ```bash
   python app.py
   ```

## 🧩 Customization
- **Fine-tuning**: Customize and fine-tune the model with your own data for improved performance and accuracy.
- **Extendable**: Add additional language models or classifiers to further enhance the chatbot’s response quality.

## 🎉 Contributing
This project is open to contributions. Feel free to fork, modify, and submit pull requests to improve the chatbot’s features.

## 🔗 Connect with the Developer
[Farhan Ahmed's Portfolio](https://farhyn.com)

---

### License
This project is licensed under the MIT License, making it free to use, modify, and distribute.

**Happy chatting!** 💬🤖
