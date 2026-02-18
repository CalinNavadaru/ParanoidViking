from rest_framework import serializers

class EmailAnalysisSerializer(serializers.Serializer):
    email_body = serializers.CharField()
