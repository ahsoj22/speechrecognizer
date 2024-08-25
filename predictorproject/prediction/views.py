
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import UserResponse
from .ml_model import train_model, predict_next_word

PROMPTS = [
    "Describe your perfect day.",
    "What are your favorite hobbies?",
    "Tell me about a memorable trip you took.",
]

def index(request):
    if request.method == 'GET':
      
        if request.session.get('prompt_index', 0) >= len(PROMPTS):
            responses = UserResponse.objects.values_list('response', flat=True)
            tokenizer, max_sequence_len = train_model(list(responses))
            request.session['tokenizer'] = tokenizer
            request.session['max_sequence_len'] = max_sequence_len
            return render(request, 'index.html', {'final_prompt': True})
        else:
            current_prompt = PROMPTS[request.session.get('prompt_index', 0)]
            return render(request, 'index.html', {'prompt': current_prompt})
    else:
        return redirect('index')

def submit_prompt(request):
    if request.method == 'POST':
        current_prompt = PROMPTS[request.session.get('prompt_index', 0)]
        user_response = request.POST.get('response')
        UserResponse.objects.create(prompt=current_prompt, response=user_response)

        request.session['prompt_index'] = request.session.get('prompt_index', 0) + 1
        return redirect('index')

def predict_word(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        words = user_input.split()

        if len(words) >= 2:
            tokenizer = request.session.get('tokenizer')
            max_sequence_len = request.session.get('max_sequence_len')
            next_word = predict_next_word(tokenizer, max_sequence_len, ' '.join(words[-2:]))
            return JsonResponse({'next_word': next_word})
        return JsonResponse({'next_word': ''})
