import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.core import normalize, windows, extract_frames, video_info, Provider, ROOT
from app import server

class CoreTests(unittest.TestCase):
    def test_invalid_model_output_is_not_an_event(self):
        for raw in ['not json', '{"category":"theft"}', '[]', '{"category":"entry_exit","summary_ko":null}']:
            self.assertEqual(normalize(raw)['category'], 'uncertain')
            self.assertFalse(normalize(raw)['parse_ok'])

    def test_valid_model_output_keeps_raw(self):
        text=json.dumps({'category':'entry_exit','summary_ko':'사람이 문으로 이동','evidence':'문과 사람 위치 변화','limitations':'이동 중 가림'})
        self.assertTrue(normalize(text)['parse_ok'])
        self.assertEqual(normalize(text)['raw'], text)

    def test_bounded_windows_have_no_false_tail(self):
        self.assertEqual(windows(12,10),[(0.,10.),(10.,12)])
        self.assertEqual(windows(120,10,2),[(0.,10.),(10.,20.)])

    def test_real_video_frames_are_ordered_and_in_range(self):
        p=ROOT/'samples/meva-01-04.mp4'
        frames=extract_frames(p,0,12,3)
        self.assertEqual(len(frames),3)
        self.assertEqual(sorted(t for t,_ in frames),[t for t,_ in frames])
        self.assertTrue(all(0<=t<12.05 for t,_ in frames))

    def test_ollama_receives_pixels_not_labels(self):
        response=type('R',(),{'raise_for_status':lambda self:None,'json':lambda self:{'message':{'content':'{"category":"uncertain","summary_ko":"작은 물체로 판단 어려움"}'}}})()
        with patch('httpx.Client') as factory:
            client=factory.return_value.__enter__.return_value
            client.post.return_value=response
            out=Provider('ollama','test-vision').analyze([(1.,Image.new('RGB',(32,32)))])
            body=client.post.call_args.kwargs['json']
            self.assertEqual(len(body['messages'][0]['images']),1)
            self.assertNotIn('ground_truth',json.dumps(body))
            self.assertEqual(out['category'],'uncertain')

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.old=server.RESULTS
        server.RESULTS=Path(self.tmp.name)
        self.client=TestClient(server.app)
    def tearDown(self):
        server.RESULTS=self.old
        self.tmp.cleanup()
    def test_cloud_requires_explicit_switch_and_key(self):
        with patch.dict(os.environ,{},clear=True):
            r=self.client.post('/api/analyze',json={'clip_id':'meva-01-04','provider':'openai'})
            self.assertEqual(r.status_code,400)
            r=self.client.post('/api/analyze',json={'clip_id':'meva-01-04','provider':'openai','allow_cloud':True})
            self.assertEqual(r.status_code,400)
    def test_cross_origin_write_rejected(self):
        r=self.client.post('/api/analyze',headers={'Origin':'https://unrelated.example'},json={'clip_id':'meva-01-04'})
        self.assertEqual(r.status_code,403)
    def test_media_byte_range(self):
        r=self.client.get('/api/video/meva-01-04',headers={'Range':'bytes=0-99'})
        self.assertEqual(r.status_code,206)
        self.assertEqual(len(r.content),100)
    def test_invalid_upload_rejected(self):
        r=self.client.post('/api/upload',files={'file':('bad.mp4',b'not video','video/mp4')})
        self.assertEqual(r.status_code,400)
    def test_valid_upload_transcodes_and_plays(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder, patch.object(server,'DATA',Path(folder)):
            with (ROOT/'samples/meva-01-04.mp4').open('rb') as f:
                response=self.client.post('/api/upload',files={'file':('door.mp4',f,'video/mp4')})
            self.assertEqual(response.status_code,200,response.text)
            clip=response.json()
            self.assertAlmostEqual(video_info(ROOT/clip['file'])['duration'],12,places=1)
            media=self.client.get('/api/video/'+clip['id'],headers={'Range':'bytes=0-99'})
            self.assertEqual(media.status_code,206)
            self.assertEqual(len(media.content),100)
    def test_analysis_lifecycle_review_and_export(self):
        fake={'category':'uncertain','summary_ko':'테스트 더블','raw':'MOCK — TEST ONLY','evidence':'','limitations':'','provider':'ollama','model':'mock','inference_seconds':0,'frame_times':[2,6,10],'review':'unreviewed'}
        with patch.object(server,'Provider') as provider:
            provider.return_value.analyze.return_value=fake
            r=self.client.post('/api/analyze',json={'clip_id':'meva-01-04','provider':'ollama','window_seconds':12})
            self.assertEqual(r.status_code,200);jid=r.json()['id']
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                data=self.client.get('/api/jobs/'+jid).json()
                if data['status']!='running':break
                time.sleep(.05)
            self.assertEqual(data['status'],'completed')
            r=self.client.post(f'/api/jobs/{jid}/segments/0/review',json={'verdict':'incorrect','note':'test'})
            self.assertEqual(r.status_code,200)
            self.assertEqual(self.client.get('/api/export/'+jid).json()['segments'][0]['review'],'incorrect')
    def test_provider_failure_is_not_completed(self):
        with patch.object(server,'Provider',side_effect=ConnectionError('private diagnostic')):
            r=self.client.post('/api/analyze',json={'clip_id':'meva-01-04'})
            jid=r.json()['id']
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                data=self.client.get('/api/jobs/'+jid).json()
                if data['status']!='running':break
                time.sleep(.05)
            self.assertEqual(data['status'],'failed')
            self.assertEqual(data['segments'],[])
            self.assertNotIn('private diagnostic',data['error'])

if __name__=='__main__':unittest.main()
