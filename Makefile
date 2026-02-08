.PHONY: setup backfill compute serve daily clean

setup:
	pip install -r requirements.txt
	python database/db.py

backfill:
	python data_collection/google_trends.py
	python data_collection/contest_scraper.py
	python data_collection/stock_data.py

compute:
	python indicator/lai_calculator.py

backtest:
	python backtesting/backtest.py

serve:
	python dashboard/app.py

daily:
	python data_collection/daily_snapshot.py

clean:
	rm -f data/lai.db
