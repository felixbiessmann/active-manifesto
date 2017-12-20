import itertools
import json
import urllib
import urllib.request

import pandas as pd


class ManifestoDataLoader(object):
    def __init__(self, api_key):
        self.base_url = "https://manifesto-project.wzb.eu/tools"
        self.country = "Germany"
        self.version = "MPDS2017b"
        self.api_key = api_key

        self.label2rightleft = {
            'right': [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606],
            'left': [103, 105, 106, 107, 403, 404, 406, 412, 413, 504, 506, 701, 202]
        }

    def cmp_code_2_left_right_neutral(self, cmp_code):
        if cmp_code in self.label2rightleft['left']:
            return 'left'
        elif cmp_code in self.label2rightleft['right']:
            return 'right'
        else:
            return 'neutral'

    @staticmethod
    def get_url(url):
        return urllib.request.urlopen(url).read().decode()

    def get_latest_version(self):
        """
        Get the latest version id of the Corpus
        """
        versions_url = self.base_url + "/api_list_metadata_versions.json?&api_key=" + self.api_key
        versions = json.loads(self.get_url(versions_url))
        return versions['versions'][-1]

    def get_manifesto_id(self, text_id, version):
        """
        Get manifesto id of a text given the text id and a version id
        """
        text_key_url = self.base_url + "/api_metadata?keys[]=" + text_id + "&version=" + version + "&api_key=" + self.api_key
        text_meta_data = json.loads(self.get_url(text_key_url))
        return text_meta_data['items'][0]['manifesto_id']

    def get_core(self):
        """
        Downloads core data set, including information about all parties
        https://manifestoproject.wzb.eu/information/documents/api
        """
        url = self.base_url + "/api_get_core?key=" + self.version + "&api_key=" + self.api_key
        return json.loads(self.get_url(url))

    def get_text_keys(self):
        d = self.get_core()
        return [p[5:7] for p in d if p[1] == self.country]

    def get_text(self, text_id):
        """
        Retrieves the latest version of the manifesto text with corresponding labels
        """
        # get the latest version of this text
        version = self.get_latest_version()
        # get the text metadata and manifesto ID
        manifesto_id = self.get_manifesto_id(text_id, version)
        text_url = self.base_url + "/api_texts_and_annotations.json?keys[]=" + manifesto_id + "&version=" + version + "&api_key=" + self.api_key
        text_data = json.loads(self.get_url(text_url))
        try:
            text = [(t['cmp_code'], t['text']) for t in text_data['items'][0]['items']]
            print('Downloaded %d texts for %s' % (len(text_data['items'][0]['items']), text_id))
            return text
        except:
            print('Could not get text %s' % text_id)

    def get_texts_per_party(self):
        # get all tuples of party/date corresponding to a manifesto text in this country
        text_keys = self.get_text_keys()
        # get the texts
        texts = {t[1] + "_" + t[0]: self.get_text(t[1] + "_" + t[0]) for t in text_keys}
        texts = {k: v for k, v in texts.items() if v}
        print("Downloaded %d/%d annotated texts" % (len(texts), len(text_keys)))
        return texts

    def get_texts(self):
        texts = self.get_texts_per_party()
        return [x for x in list(itertools.chain(*texts.values())) if x[0] != 'NA' and x[0] != '0']

    def get_manifesto_texts(self, min_len=10):
        print("Downloading texts from manifestoproject.")
        manifesto_texts = self.get_texts()
        df = pd.DataFrame(manifesto_texts, columns=['cmp_code', 'content'])
        df = df[df.content.apply(lambda x: len(str(x)) > min_len)]
        return df['content'].map(str).tolist(), df['cmp_code'].map(int).map(self.cmp_code_2_left_right_neutral).tolist()


if __name__ == "__main__":
    api_key = ""
    loader = ManifestoDataLoader(api_key)
    text, code = loader.get_manifesto_texts()
