import re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import EmailAnalysisSerializer
from .classifier import msg_clf, url_clf
from .utils import preprocess_message, preprocess_url

class AnalyzeEmailView(APIView):
    def post(self, request):
        # 1) Validate input
        serializer = EmailAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        body = serializer.validated_data['email_body']
        print(f"Body: {body}")
        # 2) Preprocess and predict on message body
        seq, length = preprocess_message(body)
        msg_label, msg_conf = msg_clf.predict((seq, length))

        # 3) Extract URLs and predict each
        urls = re.findall(r'https?://\S+', body)
        print(urls)
        url_results = []
        for url in urls:
            seq_url = preprocess_url(url)
            url_label, url_conf = url_clf.predict(seq_url)
            url_results.append({
                'url': url,
                'label': url_label,
                'confidence': url_conf
            })

        # 4) Compute overall email label: phishing if message or any URL flagged
        all_labels = [msg_label] + [u['label'] for u in url_results]
        all_confs = [msg_conf] + [u['confidence'] for u in url_results]
        email_label = 1 if any(l == 1 for l in all_labels) else 0

        # 5) Construct response
        return Response({
            'message_analysis': {
                'label': msg_label,
                'confidence': msg_conf
            },
            'url_analysis': url_results,
            'email_analysis': {
                'label': email_label,
            }
        }, status=status.HTTP_200_OK)
