from data.streams import make_stream
from mock.make_mock_results import make_mock
from eval.metrics import load_results, summary_table

df, drifts = make_stream("sudden", 0)
print(
    "stream shape:", df.shape, "| drift starts at:", drifts
)  # expect (12000, 13), [30]

make_mock()
tab = summary_table(load_results("results/mock"))
print(tab[tab.stream == "sudden"].round(2).to_string(index=False))
