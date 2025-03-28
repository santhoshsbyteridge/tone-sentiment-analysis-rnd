import boto3
import time

def analyze_audio_sentiment(audio_file_path, media_insights_pipeline_arn):
    """
    Analyzes the sentiment of an audio file using Amazon Chime SDK Media Insights Pipelines.

    Args:
        audio_file_path (str): Path to the input audio file.
        media_insights_pipeline_arn (str): ARN of the configured Media Insights Pipeline.

    Returns:
        dict: Sentiment analysis results or None if an error occurs.
    """

    chime_sdk_media_insights = boto3.client('chime-sdk-media-pipelines')
    s3 = boto3.client('s3')

    # # Upload the audio file to S3
    bucket_name = 'byteridge.cdn'  # Replace with your S3 bucket name
    object_key = f'audiochime/{time.time()}-{audio_file_path.split("/")[-1]}' #generates unique name
    try:
        s3.upload_file(audio_file_path, bucket_name, object_key)
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return None

    # Start the Media Insights Pipeline
    try:
        response = chime_sdk_media_insights.create_media_insights_pipeline(
            MediaInsightsPipelineConfigurationArn=media_insights_pipeline_arn,
            MediaInsightsPipelineName='audio-sentiment-analysis',
            S3RecordingSinkConfiguration={
                'DestinationS3Uri': f's3://{bucket_name}/results/'
            },
            KinesisDataStreamSinkConfiguration = {
                'RealTimeDataProcessingKinesisStreamConfiguration': {
                    'KinesisStreamArn':'arn:aws:kinesis:your-region:your-account-id:stream/your-kinesis-stream'
                }
            },
            SourceConfiguration={
                'S3SourceConfiguration': {
                    'S3BucketName': bucket_name,
                    'S3Key': 'audiochime/1742899494.977411-angry-speech.mp3'
                }
            }
        )

        pipeline_id = response['MediaInsightsPipeline']['MediaInsightsPipelineId']

        print(f"Media Insights Pipeline started with ID: {pipeline_id}")

        # Wait for the pipeline to complete (you'll need to poll or use event notifications in a production environment)
        time.sleep(60) #adjust time as needed, or use a better polling method.

        # Retrieve the results from S3 (this part requires further development to parse the output)
        #This assumes the output goes to the s3 bucket defined above.
        results_object_key = f'chimeresults/{pipeline_id}.json' # This may vary based on the configuration of the pipeline.
        try:
           result_response = s3.get_object(Bucket=bucket_name, Key=results_object_key)
           results_content = result_response['Body'].read().decode('utf-8')
           import json
           results = json.loads(results_content)
           return results

        except Exception as e:
            print(f"Error retrieving results from S3: {e}")
            return None

    except Exception as e:
        print(f"Error starting Media Insights Pipeline: {e}")
        return None

# Example usage:
audio_file_path = 'angry-speech.mp3' #replace with your audio file.
media_insights_pipeline_arn = 'arn:aws:chime:your-region:your-account-id:media-insights-pipeline/your-pipeline-id' #replace with your pipeline ARN

results = analyze_audio_sentiment(audio_file_path, media_insights_pipeline_arn)

if results:
    print("Sentiment Analysis Results:")
    print(results)
else:
    print("Sentiment analysis failed.")