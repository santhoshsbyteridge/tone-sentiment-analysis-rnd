import boto3
import time
import os

def analyze_voice_tone(audio_file_path, language_code='en-US', sample_rate=16000):
    """
    Analyzes voice tone (sentiment) in an audio file using Amazon Chime SDK voice analytics.

    Args:
        audio_file_path (str): Path to the audio file.
        language_code (str): Language code for the audio (e.g., 'en-US').
        sample_rate(int): sample rate of the audio file in Hz.

    Returns:
        dict: Voice tone analysis results or None if an error occurred.
    """
    chime_sdk_voice = boto3.client('chime-sdk-voice')
    s3 = boto3.client('s3')

    job_name = f"voice-tone-analysis-{int(time.time())}"
    s3_bucket = 'byteridge.cdn'  # Ensure you set this environment variable
    s3_key = f'audiochime/{time.time()}-{audio_file_path.split("/")[-1]}'

    # Upload audio to S3
    try:
        s3.upload_file(audio_file_path, s3_bucket, s3_key)
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

    media_uri = f"s3://{s3_bucket}/{s3_key}"

    try:
        response = chime_sdk_voice.create_voice_profile_domain(
            Name=f"voice-tone-domain-{int(time.time())}",
            Description="Domain for voice tone analysis",
            ServerSideEncryptionConfiguration={
                'KmsKeyArn': 'arn:aws:kms:us-east-1:123456789012:key/your-kms-key' # Replace with your KMS key ARN.
            }
        )
        print('create_voice_profile_domain response - ', response)
        domain_id = response["VoiceProfileDomain"]["VoiceProfileDomainId"]

        response = chime_sdk_voice.start_voice_tone_analysis_task(
            VoiceProfileDomainId=domain_id,
            CallDetails={
                'Participants': [
                    {
                        'ParticipantId': "participant-1",
                        'MediaInsightsPipelineConfiguration': {
                            'MediaInsightsPipelineArn': 'arn:aws:chime:us-east-1:123456789012:media-insights-pipeline/your-media-insights-pipeline' # Replace with your media insights pipeline ARN.
                        }
                    }
                ]
            },
            CallId=f"call-{int(time.time())}",
            LanguageCode=language_code,
            MediaSampleRate=sample_rate,
            S3RecordingConfig={
                'S3Uri': media_uri
            }
        )

        task_id = response['VoiceToneAnalysisTaskId']

        while True:
            status = chime_sdk_voice.get_voice_tone_analysis_task(
                VoiceToneAnalysisTaskId=task_id,
                VoiceProfileDomainId=domain_id
            )

            if status['VoiceToneAnalysisTask']['VoiceToneAnalysisTaskStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)

        if status['VoiceToneAnalysisTask']['VoiceToneAnalysisTaskStatus'] == 'COMPLETED':
            # Extract and process results from status
            results = status['VoiceToneAnalysisTask']['Results']
            chime_sdk_voice.delete_voice_profile_domain(VoiceProfileDomainId=domain_id)
            return results
        else:
            print(f"Voice tone analysis failed: {status}")
            chime_sdk_voice.delete_voice_profile_domain(VoiceProfileDomainId=domain_id)
            return None

    except Exception as e:
        print(f"Error during voice tone analysis: {e}")
        try:
          chime_sdk_voice.delete_voice_profile_domain(VoiceProfileDomainId=domain_id)
        except:
          pass
        return None

if __name__ == "__main__":
    audio_file = "sample_audio.wav"  # Replace with your audio file path
    s3_bucket_name = "your-s3-bucket-name" #Replace with your bucket name.
    os.environ["S3_BUCKET_NAME"] = s3_bucket_name

    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
    else:
        results = analyze_voice_tone(audio_file)
        if results:
            print("Voice Tone Analysis Results:")
            print(results) # Print the results. You will need to parse the results to get the specific sentiment information.
        else:
            print("Voice tone analysis failed.")