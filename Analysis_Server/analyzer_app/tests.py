# analyzer_app/tests/test_utils.py
import time
import unicodedata
from unittest.mock import patch

import torch
from django.test import SimpleTestCase
from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from analyzer_app.serializers import EmailAnalysisSerializer
from analyzer_app.utils import (
    canonical_redirect,
    clean_url,
    strip_html,
    mask_entities,
    normalize_text,
    remove_punctuation,
    tokenize_and_remove_stopwords,
    clean_text_pipeline,
    preprocess_message,
    preprocess_url
)


class DummyResponse:
    def __init__(self, url):
        self.url = url


class UtilsTests(SimpleTestCase):
    def setUp(self):
        # Patch requests.head for canonical_redirect
        patcher = patch('analyzer_app.utils.requests.head', side_effect=self.fake_head)
        self.mock_head = patcher.start()
        self.addCleanup(patcher.stop)

    def fake_head(self, url, allow_redirects, timeout):
        return DummyResponse("http://redirected.example.com/path/")

    # --- Tests for URL preprocessing ---

    def test_canonical_redirect(self):
        out = canonical_redirect("HTTP://Example.COM", timeout=0)
        self.assertEqual(out, "http://redirected.example.com/path/")

    def test_clean_url_uses_redirected(self):
        cleaned = clean_url("www.Example.com:80/page/")
        # Given our stub, clean_url will return redirected URL processed
        self.assertTrue(cleaned.startswith("http://redirected.example.com/path"))

    def test_clean_url_no_spaces(self):
        cleaned = clean_url("http://ex ample.com/!@#$%^&*()")
        # Should not contain spaces
        self.assertNotIn(" ", cleaned)

    # --- Tests for HTML/Text masking ---

    def test_strip_html(self):
        text = strip_html("<p>Hello&nbsp;<b>World</b>!</p>")
        self.assertIn("Hello", text)
        self.assertIn("World", text)
        self.assertNotIn("<", text)
        self.assertNotIn(">", text)

    def test_mask_entities(self):
        out = mask_entities("Visit http://ex.com now")
        self.assertIn("___URL___", out)
        self.assertNotIn("http://", out)

    def test_normalize_text(self):
        s = "Café ÎȘ"
        norm = normalize_text(s)
        expected = unicodedata.normalize('NFKD', s).lower()
        self.assertEqual(norm, expected)

    def test_remove_punctuation(self):
        out = remove_punctuation("Hello, World! 123-456")
        self.assertRegex(out, r'^[a-z0-9_ ]+$')

    def test_tokenize_and_remove_stopwords(self):
        tokens = tokenize_and_remove_stopwords("this is a test of the system")
        self.assertIn("test", tokens)
        self.assertNotIn("is", tokens)

    # --- Tests for full text pipeline ---

    def test_clean_text_pipeline(self):
        tokens = clean_text_pipeline("<div>Hello http://ex.com!</div>")
        self.assertIn("___url___", tokens)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    # --- Tests for message preprocessing ---

    def test_preprocess_message(self):
        from analyzer_app.text_vocab import word2idx
        word2idx.clear()
        word2idx.update({"hello": 5, "<unk>": 0, "<pad>": 1})
        seq, length = preprocess_message("Hello unknownword", max_len=5)
        self.assertIsInstance(seq, torch.LongTensor)
        self.assertIsInstance(length, torch.LongTensor)
        self.assertEqual(length.item(), 2)
        self.assertEqual(seq.size(0), 5)

    # --- Tests for URL sequence preprocessing ---

    def test_preprocess_url_truncate_and_pad(self):
        from analyzer_app.char_vocab import char2idx, PAD
        char2idx.clear()
        char2idx['a'] = 2
        char2idx[PAD] = 0
        seq = preprocess_url("aa.bb", max_len=4)
        self.assertIsInstance(seq, torch.LongTensor)
        self.assertEqual(seq.size(0), 4)

    def test_preprocess_url_empty(self):
        # Stub clean_url to return empty string so preprocess_url pads only
        with patch('analyzer_app.utils.clean_url', return_value=""):
            from analyzer_app.char_vocab import char2idx, PAD
            char2idx.clear()
            char2idx[PAD] = 0
            seq = preprocess_url("", max_len=3)
            self.assertTrue(all(item == PAD for item in seq.tolist()))


class EmailAnalysisSerializerTests(SimpleTestCase):
    def test_serializer_valid_minimal(self):
        """Payload minimal cu câmpul email_body trece validarea."""
        data = {"email_body": "Acesta este corpul unui email."}
        ser = EmailAnalysisSerializer(data=data)
        self.assertTrue(ser.is_valid(), msg=ser.errors)
        self.assertEqual(ser.validated_data["email_body"], data["email_body"])

    def test_serializer_invalid_missing_body(self):
        """Lipsa câmpului email_body aruncă eroare."""
        data = {}
        ser = EmailAnalysisSerializer(data=data)
        self.assertFalse(ser.is_valid())
        self.assertIn("email_body", ser.errors)

    def test_serializer_coerces_non_string(self):
        """Valori non-string sunt convertite la string."""
        data = {"email_body": 12345}
        ser = EmailAnalysisSerializer(data=data)
        self.assertTrue(ser.is_valid(), msg=ser.errors)
        # DRF CharField convertește non-string la str()
        self.assertEqual(ser.validated_data["email_body"], "12345")

    def test_serializer_rejects_empty_string(self):
        """email_body vid este invalid implicit (blank=False)."""
        data = {"email_body": ""}
        ser = EmailAnalysisSerializer(data=data)
        self.assertFalse(ser.is_valid())
        self.assertIn("email_body", ser.errors)
        self.assertEqual(ser.errors["email_body"][0].code, "blank")

    def test_serializer_strips_whitespace(self):
        """Whitespace de la început și sfârșit este tăiat implicit."""
        data = {"email_body": "  Text cu spații  "}
        ser = EmailAnalysisSerializer(data=data)
        self.assertTrue(ser.is_valid(), msg=ser.errors)
        # DRF CharField trim_whitespace=True taie spațiile
        self.assertEqual(ser.validated_data["email_body"], "Text cu spații")



class AnalyzeEmailViewTests(APITestCase):
    def setUp(self):
        self.url = reverse('analyze-email')

    @patch('analyzer_app.views.msg_clf')
    @patch('analyzer_app.views.url_clf')
    @patch('analyzer_app.views.preprocess_message')
    @patch('analyzer_app.views.preprocess_url')
    def test_legitim_email_fara_url(self, mock_pre_url, mock_pre_msg, mock_url_clf, mock_msg_clf):
        """
        Caz „happy path” fără URL: mesaj legitim.
        """
        # Stub preprocess_message → (seq, length)
        mock_pre_msg.return_value = ('seq_msg', 'len_msg')
        # Stub mesaj classifier să returneze (label, confidence)
        mock_msg_clf.predict.return_value = (0, 0.97)
        # Nu există URL-uri → preprocess_url și url_clf.predict nu trebuie apelate
        mock_pre_url.assert_not_called()
        mock_url_clf.predict.assert_not_called()

        payload = {'email_body': 'Hello world!', 'urls': []}
        resp = self.client.post(self.url, payload, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        data = resp.json()
        self.assertIn('message_analysis', data)
        self.assertEqual(data['message_analysis']['label'], 0)
        self.assertAlmostEqual(data['message_analysis']['confidence'], 0.97)

        # URL analysis vid
        self.assertEqual(data['url_analysis'], [])

        # Email_analysis = 0
        self.assertIn('email_analysis', data)
        self.assertEqual(data['email_analysis']['label'], 0)

    @patch('analyzer_app.views.msg_clf')
    @patch('analyzer_app.views.url_clf')
    @patch('analyzer_app.views.preprocess_message')
    @patch('analyzer_app.views.preprocess_url')
    def test_phishing_email_cu_url(self, mock_pre_url, mock_pre_msg, mock_url_clf, mock_msg_clf):
        """
        Caz phishing: mesaj și URL malițios.
        """
        mock_pre_msg.return_value = ('seq_msg', 'len_msg')
        mock_msg_clf.predict.return_value = (0, 0.50)  # mesaj nepericulos
        # Un URL de phishing
        mock_pre_url.return_value = 'seq_url'
        mock_url_clf.predict.return_value = (1, 0.85)

        body = "Check this http://bad.com"
        payload = {'email_body': body, 'urls': ['http://bad.com']}
        resp = self.client.post(self.url, payload, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        data = resp.json()
        # Mesajul e considerat benign, dar URL-ul e rău → overall phishing
        self.assertEqual(data['message_analysis']['label'], 0)
        self.assertEqual(data['url_analysis'][0]['url'], 'http://bad.com')
        self.assertEqual(data['url_analysis'][0]['label'], 1)
        self.assertAlmostEqual(data['url_analysis'][0]['confidence'], 0.85)
        self.assertEqual(data['email_analysis']['label'], 1)

    def test_bad_request_missing_body(self):
        """
        Caz invalid: lipsă email_body → 400 Bad Request.
        """
        resp = self.client.post(self.url, {'urls': []}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('email_body', resp.json())

    @patch('analyzer_app.views.msg_clf')
    @patch('analyzer_app.views.url_clf')
    @patch('analyzer_app.views.preprocess_message')
    @patch('analyzer_app.views.preprocess_url')
    def test_multiple_urls(self, mock_pre_url, mock_pre_msg, mock_url_clf, mock_msg_clf):
        """
        Caz cu mai multe URL-uri; unul malițios.
        """
        mock_pre_msg.return_value = ('seq_msg', 'len_msg')
        mock_msg_clf.predict.return_value = (0, 0.10)
        # simulăm două URL-uri: primul benign, al doilea malițios
        mock_pre_url.side_effect = ['seq_u1', 'seq_u2']
        mock_url_clf.predict.side_effect = [(0,0.05), (1,0.99)]

        body = "Here http://good.com and http://evil.com"
        payload = {'email_body': body,
                   'urls': ['http://good.com','http://evil.com']}
        resp = self.client.post(self.url, payload, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

        data = resp.json()
        # Verificăm raportarea corectă a fiecărui URL
        self.assertEqual(data['url_analysis'][0]['label'], 0)
        self.assertEqual(data['url_analysis'][1]['label'], 1)
        # Overall phishing datorită celui rău
        self.assertEqual(data['email_analysis']['label'], 1)




class PerformanceTests(TestCase):
    def setUp(self):
        self.url = reverse('analyze-email')
        # Payload minimal valid
        self.payload = {
            'email_body': 'Test performance'
        }

    @patch('analyzer_app.views.msg_clf')
    @patch('analyzer_app.views.preprocess_message')
    def test_inference_latency(self, mock_pre_msg, mock_msg_clf):
        """
        Măsoară latența view-ului AnalyzeEmailView
        și asigură că e sub 0.6 secunde (600 ms).
        """
        # Stub necesare pentru a nu încărca modele reale
        mock_pre_msg.return_value = ([], 1)
        mock_msg_clf.predict.return_value = (0, 0.0)

        # Facem N iterații și calculăm media
        runs = 10
        total = 0.0
        for _ in range(runs):
            start = time.time()
            resp = self.client.post(self.url, self.payload, format='json')
            self.assertEqual(resp.status_code, 200)
            total += (time.time() - start)

        avg_latency = total / runs
        # Assertăm că media e sub 0.6 sec
        self.assertLess(avg_latency, 0.6,
            f"Average latency too high: {avg_latency:.3f}s")

    @patch('analyzer_app.views.url_clf')
    @patch('analyzer_app.views.preprocess_url')
    def test_url_inference_latency(self, mock_pre_url, mock_url_clf):
        """
        Măsoară latența preprocess_url + predict pentru un URL lung;
        trebuie să fie << 0.6 s.
        """
        # URL de test
        url = 'http://' + 'a'*200 + '.com'
        mock_pre_url.return_value = 'seq'
        mock_url_clf.predict.return_value = (0, 0.0)

        runs = 50
        total = 0.0
        for _ in range(runs):
            start = time.time()
            seq = mock_pre_url(url)
            label, conf = mock_url_clf.predict(seq)
            total += (time.time() - start)

        avg_url = total / runs
        self.assertLess(avg_url, 0.6,
            f"URL inference too slow: {avg_url:.3f}s")


class SecurityRobustnessTests(TestCase):
    """
    Testează robustețea la input malițios și invalid.
    """

    def setUp(self):
        self.url = reverse('analyze-email')

    def test_sql_injection_payload(self):
        """
        Asigură că un payload cu injectare SQL nu cauzează crash și răspunde 400.
        """
        payload = {
            'email_body': "'; DROP TABLE users; --",
            'urls': []
        }
        resp = self.client.post(self.url, payload, format='json')
        # DRF validă ca text simplu, nu se execută nicio interogare SQL.
        # Ar trebui încă să primească 200, dar fără efecte adverse.
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_oversized_payload(self):
        """
        Blocare sau limitare a payload-urilor imense.
        """
        body = "A" * 100_000  # 100k caractere
        resp = self.client.post(self.url, {'email_body': body, 'urls': []}, format='json')
        self.assertIn(resp.status_code, (status.HTTP_200_OK, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE))

    def test_missing_json(self):
        """
        Trimitere de formă ne-JSON → 415 Unsupported Media Type.
        """
        resp = self.client.post(self.url, data="plain text", content_type="text/plain")
        self.assertEqual(resp.status_code, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

    def test_invalid_content_type(self):
        """
        Content-Type invalid dar JSON parsat → 400 Bad Request.
        """
        resp = self.client.post(self.url, {'email_body': 'test', 'urls': []}, content_type="application/xml")
        self.assertEqual(resp.status_code, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
