## CareSetu – AI-Powered Baby Care Companion 👶💖
CareSetu is a smart, mobile-friendly AI application designed to support Indian parents through the crucial first 1000 days of their baby's life. Leveraging the power of AI, this tool aims to bridge the information gap, especially for parents in rural and low-income communities, by providing accessible and reliable childcare guidance.

## 💡 Key Features
🤖 AI Parenting Assistant: Get instant, personalized answers to parenting questions (e.g., "What to do for a baby's fever?") powered by a robust AI model.

📅 Vaccine Tracker: Automatically generates a complete vaccination schedule based on your baby's date of birth, ensuring you never miss a crucial immunization.

📷 Visual Health Log: Upload photos of common concerns like skin rashes, eye redness, or tongue spots for a preliminary AI-based suggestion.

🏥 Nearby Hospital Finder: Quickly locate nearby government and private baby care hospitals based on your PIN code.

📊 Growth Tracker: Monitor your baby's development by tracking their height and weight against standard growth metrics.

📖 1000 Days Guide: An in-app guide with essential information on developmental milestones, nutrition, and safe practices.

🌐 Multilingual Support: Designed to provide responses in multiple Indian languages, starting with Hindi, to ensure wider accessibility.

## 🛠️ Tech Stack
Frontend: Streamlit

Backend: Python

AI/ML: Placeholder logic for Large Language Models (like IBM Granite or Google Gemini)

Core Libraries: Pillow, Pandas
# Video link-> 
https://www.youtube.com/watch?v=mwKiuFczScw

## 🖥️ Setup and Installation
Follow these steps to run the application on your local machine.

Clone the Repository

Bash

git clone https://github.com/your-username/CareSetu.git
cd CareSetu
Create a Virtual Environment (Recommended)

Bash

python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate # On macOS/Linux
Install Dependencies
Make sure you have a requirements.txt file with all the necessary libraries.

Bash

pip install -r requirements.txt
Run the Application

Bash

streamlit run src/app.py
## 🚀 Future Improvements
-Live API Integration: Connect the app to a live LLM API (Google Gemini, OpenAI, etc.) for real-time AI responses.
-Computer Vision Model: Enhance the Health Log with a trained vision model to identify common baby health issues from images.

-User Authentication & Data Persistence: Add user accounts to save and track a child's growth and vaccination history over time.

-WhatsApp Integration: Develop a WhatsApp bot for even greater accessibility among users without consistent internet access.
