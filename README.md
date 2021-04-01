# wav2vec2

Hugging Face has orginized [Fine Tuning Week](https://discuss.huggingface.co/t/open-to-the-community-xlsr-wav2vec2-fine-tuning-week-for-low-resource-languages/) for the model named [Wav2Vec2 XLSR](https://huggingface.co/facebook/wav2vec2-large-xlsr-53). During the week, I have fine tuned this model on Turkish and Finnish by using the [Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset. The models I have shared on the [Hugging Face Hub](https://huggingface.co/dundar/) are **1st placed** in each language. Leaderboards are seen in the webside of Paperswithcode for [Turkish](https://paperswithcode.com/sota/speech-recognition-on-common-voice-turkish) and [Lithuanian](https://paperswithcode.com/sota/speech-recognition-on-common-voice-lithuanian).


## Fine-tuning for Turkish [HuggingFace Hub](https://huggingface.co/dundar/wav2vec2-large-xlsr-53-turkish)
* Before executing the training code, the following packages should be installed.

> pip install datasets==1.4.1  
> pip install transformers==4.4.0  
> pip install torchaudio  
> pip install librosa  
> pip install jiwer  

* Run the following command to start the training process.
> python finetune.py

* The same code is also utilized to continue the training from checkpoints to reach the step 3200.

## Fine-tuning for Lithuanian [HuggingFace Hub](https://huggingface.co/dundar/wav2vec2-large-xlsr-53-lithuanian)

I have used the notebook [here](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb) with the following parameters for the class named Wav2Vec2ForCTC.

    activation_dropout=0.055,
    attention_dropout=0.094,
    hidden_dropout=0.047,
    feat_proj_dropout=0.04,
    mask_time_prob=0.082,
    layerdrop=0.041,
    gradient_checkpointing=True, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
    
Training arguments are the following:

    group_by_length=True, 
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=2, 
    evaluation_strategy="steps", 
    num_train_epochs=30, 
    fp16=True, 
    save_steps=400, 
    eval_steps=400, 
    logging_steps=400, 
    learning_rate=2.34e-4, 
    warmup_steps=500, 
    save_total_limit=2, 
