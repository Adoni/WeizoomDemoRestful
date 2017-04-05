#coding:utf8
from utils import get_database
from utils import get_qianyue_database
from utils import get_ltp_path
import sys
from pyltp import Segmentor
from preprocessing import tokenizer
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import random
from texttable import Texttable
import numpy

wechat_db = get_database(sys.argv[1]).neaten_wechat2
account_info_db = get_database(sys.argv[1]).wechat_old_account_info


def get_data_of_an_account(account_id):
    data = wechat_db.find_one({'_id': account_id})
    if data is None:
        return data
    data['articles'], data['articles2'] = [], data['articles']
    for article in data['articles2']:
        #if article['position'] == '头条':
            data['articles'].append(article)
    return data


def get_account_ids(account_name):
    accounts = account_info_db.find({'name': account_name})
    if accounts == None or accounts == []:
        return None
    else:
        account_ids = [account['str_id'] for account in accounts]
        return account_ids


not_interested_category_1 = ['生活服务', '农资采购', '更多服务', '']
not_interested_category_1 = set(not_interested_category_1)


class ScoreProvider_V1:
    def __init__(self,config_file):
        self.segmentor = Segmentor()
        print('Loading LTP data')
        self.segmentor.load_with_lexicon(
            '%s/cws.model' % (get_ltp_path(config_file)),
            '/Users/sunxiaofei/workspace/WeizoomData/3rd_categories.data')
        self.all_3rd_category_key_words = self.get_all_3rd_category_key_words(config_file)

        print('Loading content transformer')
        self.content_vectorizer = pickle.load(
            open('./wechat_content_vectorizer.pkl', 'rb'))
        self.content_transformer = pickle.load(
            open('./wechat_content_transformer.pkl', 'rb'))
        self.content_features = self.content_vectorizer.get_feature_names()

        print('Loading information transformer')
        self.info_vectorizer = pickle.load(
            open('./wechat_info_vectorizer.pkl', 'rb'))
        self.info_transformer = pickle.load(
            open('./wechat_info_transformer.pkl', 'rb'))
        self.info_features = self.info_vectorizer.get_feature_names()

        print('Done')

    def get_all_3rd_category_key_words(self):
        db = get_qianyue_database(config_name).tf_idf_normalizition_rank_1title
        db = get_qianyue_database(config_name).tf_tdf_normalizition_rank_1title
        key_words = dict()
        for category in db.find():
            if category['catalog1'] in not_interested_category_1:
                continue
            categories = [
                category['catalog1'],
                category['catalog2'],
                category['catalog3']
            ]
            key_words['||||'.join(categories)] = list(
                map(lambda w: (' '.join(w.split(':')[0:-1]), float(w.split(':')[-1])),
                    category['words'][:50]))
        print(len(key_words))
        return key_words

    def get_category_score_from_content(self, account_data):
        scores = dict()
        document = []
        for article in account_data['articles']:
            for sentence in article['content']:
                document += sentence
        tf_idf = self.content_transformer.transform(
            self.content_vectorizer.transform([document])).toarray()[0]
        tf_idf_dict = dict(
            filter(lambda x: x[1] > 0, zip(self.content_features, list(
                tf_idf))))
        for k, v in self.all_3rd_category_key_words.items():
            scores[k] = []
            for word, value1 in v:
                if word in tf_idf_dict:
                    scores[k].append((word, value1, tf_idf_dict[word]))
        scores = sorted(
            scores.items(),
            key=lambda s: sum(list(map(lambda x: x[1] * x[2], s[1]))),
            reverse=True)
        return scores

    def get_category_score_from_info(self, account_data):
        scores = dict()
        document = []
        document += list(self.segmentor.segment(account_data['info']['name']))
        document += list(
            self.segmentor.segment(account_data['info']['description']))
        document += account_data['info']['tags']
        document = ' '.join(document).replace('\n', ' ').replace(
            '\r', ' ').replace('\b', ' ').split(' ')
        tf_idf = self.info_transformer.transform(
            self.info_vectorizer.transform([document])).toarray()[0]
        tf_idf_dict = dict(
            filter(lambda x: x[1] > 0, zip(self.info_features, list(tf_idf))))
        for k, v in self.all_3rd_category_key_words.items():
            scores[k] = []
            for word, value1 in v:
                if word in tf_idf_dict:
                    scores[k].append((word, value1, tf_idf_dict[word]))
        scores = sorted(
            scores.items(),
            key=lambda s: sum(map(lambda x: x[1] * x[2], s[1])),
            reverse=True)
        return scores

    def get_category_score(self, account_data, interested_position):
        content_score = self.get_category_score_from_content(account_data)
        info_score = self.get_category_score_from_info(account_data)
        dict_content_score = dict(content_score)
        dict_info_score = dict(info_score)
        scores = []
        for key in set(dict_content_score.keys()).union(
                set(dict_info_score.keys())):
            if key in dict_content_score:
                v_content = sum(
                    map(lambda x: x[1] * x[2], dict_content_score[key]))
            else:
                v_content = 0.0
            if key in dict_info_score:
                v_info = sum(map(lambda x: x[1] * x[2], dict_info_score[key]))
            else:
                v_info = 0.0
            scores.append((key, v_content, v_info, v_content + v_info))

        scores = sorted(scores, key=lambda s: s[interested_position], reverse=True)
        return scores

    def get_account_key_words_from_content(self, account_data, top=10):
        document = []
        for article in account_data['articles']:
            for sentence in article['content']:
                document += sentence
        tf_idf = self.content_transformer.transform(
            self.content_vectorizer.transform([document])).toarray()[0]
        tf_idf = filter(lambda x: x[1] > 0 and len(x[0]) > 1,
                        zip(self.content_features, tf_idf))
        key_words = sorted(tf_idf, key=lambda x: x[1], reverse=True)[:top]
        return key_words

    def get_account_key_words_from_info(self, account_data, top=10):
        document = []
        document += list(self.segmentor.segment(account_data['info']['name']))
        document += list(
            self.segmentor.segment(account_data['info']['description']))
        document += account_data['info']['tags']
        document = ' '.join(document).replace('\n', ' ').replace(
            '\r', ' ').replace('\b', ' ').split(' ')
        tf_idf = self.info_transformer.transform(
            self.info_vectorizer.transform([document])).toarray()[0]
        tf_idf = filter(lambda x: x[1] > 0 and len(x[0]) > 1,
                        zip(self.info_features, tf_idf))
        key_words = sorted(tf_idf, key=lambda x: x[1], reverse=True)[:top]
        return key_words

    def get_key_words(self,account_data,top=10):
        content_key_words = dict(sp.get_account_key_words_from_content(data, 100))
        info_key_words = dict(sp.get_account_key_words_from_info(data, 100))
        key_words=set(content_key_words.keys()+info_key_words.keys())
        data=[]
        for key in key_words:
            weight=0.0
            try:
                weight+=content_key_words[key]
            except:
                pass
            try:
                weight+=content_key_words[key]
            except:
                pass
            data.append({'word':key,'weight':weight})
        data=sorted(data, key=lambda x:x['weight'], reverse=True)[:top]
        return data

class api:
    def __init__(self, config_file):
        self.tmall_info_db=get_database(config_file).neaten_tmall
        self.wechat_info_db=get_database(config_file).neaten_wechat
        self.sp = ScoreProvider_V1(config_file)

    def get_account_info_by_id(self, wechat_id):
        account = wechat_db.find_one({'_id': wechat_id})
        if account is None:
            return {}
        data=dict()
        data['_id']=wechat_id
        data['name']=account['name']
        data['description']=account['description']
        data['articles']=[]
        for article in account['articles']:
            ar=dict()
            ar['title']=article['title']
            ar['content']=[]
            for p in account['content']:
                p=' '.join(p)
                ar['content'].append(p)
            if len(' '.join(p))<20:
                continue
            data['article'].append(ar)
        return data

    def get_account_info_by_name(self, wechat_name):
        accounts = self.wechat_info_db.find_one({'name': account_name},{'_id':1})
        return self.get_account_info_by_id(account['_id'])

    def get_key_words(self, wechat_id):
        account_data = get_data_of_an_account(wechat_id)
        if account_data == None:
            return {}
        data=dict()
        data['_id']=wechat_id
        data['keywords']=self.sp.get_key_words(account_data,20)
        return data

    def get_recommendations(self, wechat_id, ignored_key_words=[]):
        account_data = get_data_of_an_account(wechat_id)
        if account_data == None:
            return {}
        score=self.sp.get_category_score(account_data, interested_position)
        data=dict()
        data['_id']=wechat_id
        data['categories']=list
        interested_position=3
        for s in score[10]:
            if s[interested_position]==0:
                continue
            s[0]=s[0].split('||||')
            category=dict()
            category['category1']=s[0][0]
            category['category2']=s[0][1]
            category['category3']=s[0][2]
            category['weight']=s[interested_position]
            category['products']=list()
            category_data=self.tmall_info_db.find_one({'_id':'||||'.join(s[0])})
            for p in category_data['products']:
                product=dict()
                url=p['href']
                if url.startswith('//'):
                    url='https:'+url
                product['url']=url
                product['price']=p['price']
                category['products'].append(product)
                url=p['img_src']
                if url.startswith('//'):
                    url='https:'+url
                product['image_url']=url
            data['categories'].append(category)
        return data
