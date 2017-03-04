from model.models import TradeModel


def test_trade_model():
    trade_model = TradeModel('test/data/raw_data.csv', frequency='D')

    assert trade_model.df.shape[0] > 3
    print trade_model.df.head()
    assert trade_model.df.index.name == 'Time (UTC)'
