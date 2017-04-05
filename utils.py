import sys
if sys.version.startswith('3.'):
    import configparser as ConfigParser
else:
    import ConfigParser
from pymongo import MongoClient


def get_database(config_fname):
    config = ConfigParser.ConfigParser()
    config.read(config_fname)
    uri = 'mongodb://%s:%s@%s:%s/%s' % (
        config.get('database', 'username'), config.get('database', 'password'),
        config.get('database', 'host'), config.get('database', 'port'),
        config.get('database', 'dbname'))
    client = MongoClient(uri)
    db = client.get_database('weizoom')
    return db


def get_qianyue_database(config_fname):
    config = ConfigParser.ConfigParser()
    config.read(config_fname)
    uri = 'mongodb://%s:%s@%s:%s/%s' % (
        config.get('qianyue_database', 'username'),
        config.get('qianyue_database', 'password'),
        config.get('qianyue_database', 'host'),
        config.get('qianyue_database', 'port'),
        config.get('qianyue_database', 'dbname'))
    client = MongoClient(uri)
    db = client.get_database('weizoom')
    return db


def get_ltp_path(config_fname):
    config = ConfigParser.ConfigParser()
    config.read(config_fname)
    return config.get('path', 'ltp')


def get_wechat_tags():
    db = get_database('./config.ini').wechat_account_info
    tags = []
    for item in db.find():
        tags += item['tags']
    tags = set(tags)
    open('./tags.data', 'w').write('\n'.join(tags))
    print(len(tags))


if __name__ == '__main__':
    get_message_queue('../config.ini', 'hehe')
