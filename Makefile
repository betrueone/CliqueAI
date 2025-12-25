include $(ENV_FILE)

ifndef ENV_FILE
$(error ENV_FILE is not set)
endif

export PYTHONPATH=$(PWD)

register:
	{ \
		read -p 'Wallet name?: ' wallet_name ;\
		read -p 'Hotkey?: ' hotkey_name ;\
		btcli subnet register --netuid $(netuid) --wallet.name "$$wallet_name" --wallet.hotkey "$$hotkey_name" --subtensor.chain_endpoint $($(NETWORK)) ;\
	}

miner:
	pm2 start CliqueAI/miner.py --name $(MINER_NAME) --interpreter .venv/bin/python -- \
		--netuid $(NETUID) \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(HOTKEY) \
		--subtensor.network $(NETWORK) \
		--axon.ip $(AXON_IP) \
		--axon.port $(AXON_PORT) \
		--neuron.autoupdate $(AUTO_UPDATE) \
		--logging.info

miner_dev:
	PYTHONPATH=. .venv/bin/python CliqueAI/miner.py \
		--netuid $(NETUID) \
		--wallet.name $(WALLET_NAME) \
		--wallet.hotkey $(HOTKEY) \
		--subtensor.network $(NETWORK) \
		--axon.ip $(AXON_IP) \
		--axon.port $(AXON_PORT) \
		--neuron.autoupdate $(AUTO_UPDATE) \
		--logging.info