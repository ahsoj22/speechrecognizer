from django.shortcuts import render, redirect
from .models import UserResponse
from .ml_model import load_and_preprocess_text, train_model_on_dataset, fine_tune_model, predict_next_word
import json
from django.http import JsonResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_FILE_PATH = os.path.join(BASE_DIR, 'data', 'book-war-and-peace.txt')

text_lines = load_and_preprocess_text(TEXT_FILE_PATH)
# Pre-train the model when the server starts
model, tokenizer, max_sequence_len = train_model_on_dataset(text_lines)
  
PROMPTS = [
    "Describe your perfect day. (Please do not use punctuation)",
    "What are your favorite hobbies? (Please do not use punctuation)",
    "Tell me about a memorable trip you took. (Please do not use punctuation)",
    "How would you describe your favorite meal? (Please do not use punctuation)",
    "How would you describe your favorite book? (Please do not use punctuation)",
    "How would you describe yourself as a person? (Please do not use punctuation)", 
    "Tell me about your favorite animal. (Please do not use punctuation)", 
    "Tell me about your favorite color and why. (Please do not use punctuation)", 
]

def index(request):
    # Initialize session if it doesn't exist
    if 'prompt_index' not in request.session:
        request.session['prompt_index'] = 0
        request.session['responses_collected'] = False

    # If all prompts are answered and responses have not been collected yet, start training
    if request.session.get('prompt_index') >= len(PROMPTS) and not request.session.get('responses_collected'):
        # Collect user responses and fine-tune the model
        responses = UserResponse.objects.values_list('response', flat=True)
        global model
        model = fine_tune_model(model, tokenizer, list(responses), max_sequence_len)
        request.session['responses_collected'] = True
        return render(request, 'index.html', {'final_prompt': True})
    
    # If training is done and ready for prediction
    elif request.session.get('responses_collected'):
        return render(request, 'index.html', {'final_prompt': True})

    # Otherwise, show the next prompt
    current_prompt = PROMPTS[request.session['prompt_index']]
    return render(request, 'index.html', {'prompt': current_prompt})

def submit_prompt(request):
    if request.method == 'POST':
        # Save user response
        current_prompt = PROMPTS[request.session.get('prompt_index', 0)]
        user_response = request.POST.get('response')
        UserResponse.objects.create(prompt=current_prompt, response=user_response)

        # Increment the session's prompt index to show the next prompt
        request.session['prompt_index'] += 1

        return redirect('index')
    

def predict_word(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_input = data.get('user_input', '')
            if not user_input:
                return JsonResponse({'error': 'No user input provided'}, status=400)

            user_input = preprocess_input(user_input)
            words = user_input.split()

            if len(words) >= 2:
                # Modify the prediction logic to return top 3 likely words
                top_3_words = predict_top_3_words(model, tokenizer, max_sequence_len, ' '.join(words[-2:]))
                return JsonResponse({'predicted_words': top_3_words})
            else:
                return JsonResponse({'predicted_words': []})
        except Exception as e:
            print(f"Error: {str(e)}")  # Log the error to the console for debugging
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
def preprocess_input(user_input):
    if user_input is None:
        user_input = ''
    return user_input.lower()
    
def predict_top_3_words(model, tokenizer, max_sequence_len, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]

    # Get indices of top 3 predictions
    top_3_indices = predicted_probs.argsort()[-3:][::-1]

    # Map indices back to words
    top_3_words = [word for word, index in tokenizer.word_index.items() if index in top_3_indices]
        
    return top_3_words
    
def reset_session(request):
    request.session.flush()  
    return redirect('index')  