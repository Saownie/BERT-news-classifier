from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)
app.secret_key = 'my_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:password123@localhost/flask_bert'


#initialize the model 
model_path = '/Users/sabbirahamedsaown/Desktop/Final Year Project/Model' 
tokenizer = AutoTokenizer.from_pretrained(model_path) 
model = AutoModelForSequenceClassification.from_pretrained(model_path) 

# Initialize SQLAlchemy and Flask-Migrate
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Dummy users to demonstrate authorization
def create_dummy_users():
    if User.query.count() == 0:
        admin = User(username='admin', password='admin_password', role='admin')
        user1 = User(username='user1', password='user1_password', role='user')
        db.session.add(admin)
        db.session.add(user1)
        db.session.commit()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    transcription = None
    if request.method == 'POST':
        if 'news_text' in request.form and request.form['news_text'] != '':
            transcription = request.form['news_text']
            return redirect(url_for('predict', news_text=transcription))

    return render_template('index.html', transcription=transcription)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    news_text = request.args.get('news_text')
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    prediction_scores = predictions[0].tolist()
    
    #  0 is 'unreliable' and 1 is 'reliable'
    reliability_score = prediction_scores[1]  # index for 'reliable'
    if reliability_score > 0.5:
        result = f"This news is likely reliable with a confidence of {reliability_score:.2%}."
        prediction_reliable = True
    else:
        result = f"This news is likely unreliable with a confidence of {(1-reliability_score):.2%}."
        prediction_reliable = False

    return render_template('index.html', prediction_text=result, prediction_reliable=prediction_reliable, transcription=news_text)

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    feedback_text = request.form['feedback']
    satisfaction = request.form['satisfaction']
    # Here you can save the feedback to a database or process it as needed
    print(f"Received feedback: {feedback_text}, Satisfaction: {satisfaction}")
    # Save feedback to session for demonstration purposes
    if 'feedbacks' not in session:
        session['feedbacks'] = []
    session['feedbacks'].append({'feedback': feedback_text, 'satisfaction': satisfaction, 'user': current_user.id})
    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('index'))

@app.route('/admin/feedbacks')
@login_required
def view_feedbacks():
    if current_user.role != 'admin':
        flash('You do not have access to this page.', 'danger')
        return redirect(url_for('index'))
    feedbacks = session.get('feedbacks', [])
    return render_template('feedbacks.html', feedbacks=feedbacks)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_dummy_users()
    app.run(debug=True)

