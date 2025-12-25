import asyncio
import json
import os
import time
import typing
from concurrent.futures import ThreadPoolExecutor

import bittensor as bt
from CliqueAI.clique_algorithms import (
    networkx_algorithm,
)
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph
from common.base.miner import BaseMinerNeuron
from pmc.pmc import pmc_algorithm

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    WHITELISTED_VALIDATORS = {
        "5CJmEokTjtqe4bsXsNfEzTv9bjW26R4A4iYS7hi1nz5p4s83": "Rizzo(Insured)",
        "5FLoWCDovMPeH3Gv4syQSZ8TuKcMv6N27g8diDU8zJSeRv8m": "5FLoWCDovMPeH3Gv4syQSZ8TuKcMv6N27g8diDU8zJSeRv8m", 
        "5EHGayLmiXfwz6oYmQFYmDz12RPXkhJw8Ty8RYVDeVZH9Q5L": "CliqueAI",
        "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u": "tao.bot",
        "5CFZgeUAguZLv4VYt38jN6HtWAVEn3vxVmBJByHXGTeEsn83": "RoundTable21",
        "5Gsvaq5SsdJ85Eu2XWKfLrhZ2j48mez1urYZXCu9Sga2grmE": "Yuma",
        "5Fh1YDPcJ2Bs6JHfE5Ck1gCwvPxgkK32ZpHy94oy9XKMEyh9": "OpenTensor",
        "5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5": "1T1B.AI",
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        # Create a thread pool executor for running blocking clique algorithms
        # This allows concurrent request handling without blocking the async event loop
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clique_worker")
        self.axon.attach(
            forward_fn=self.forward_graph,
            blacklist_fn=self.backlist_graph,
            priority_fn=self.priority_graph,
        )

    async def forward_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> MaximumCliqueOfLambdaGraph:
        codec = GraphCodec()
        adjacency_matrix = codec.decode_matrix(synapse.encoded_matrix)
        adjacency_list = codec.matrix_to_list(adjacency_matrix)
        
        # Run blocking clique algorithms in a thread pool to avoid blocking the event loop
        # This allows the miner to handle multiple concurrent requests
        # This avoids diversity penalties when running multiple miners
        try:
            maximum_clique: list[int] = pmc_algorithm(synapse.number_of_nodes, adjacency_list)
            if len(maximum_clique) == 0:
                raise Exception("No maximum clique found with pmc algorithm")
        except Exception as e:
            maximum_clique = networkx_algorithm(
                synapse.number_of_nodes,
                adjacency_list
            )

        bt.logging.info(
            f"Maximum clique found: {maximum_clique} with size {len(maximum_clique)} (UID: {self.uid}, UUID: {synapse.uuid})"
        )
        synapse.maximum_clique = maximum_clique

        try:
            data = {
                "uuid": synapse.uuid,
                "number_of_nodes": synapse.number_of_nodes,
                "adjacency_list": adjacency_list,
                "maximum_clique": maximum_clique,
            }
            os.makedirs("data", exist_ok=True)
            with open(f"data/maximum_clique_{synapse.uuid}.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            bt.logging.error(f"Error saving maximum clique data: {e}")

        return synapse

    async def backlist_graph(
        self, synapse: MaximumCliqueOfLambdaGraph
    ) -> typing.Tuple[bool, str]:
        # Check if the request has a valid dendrite and hotkey
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.WHITELISTED_VALIDATORS:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # If all checks pass, allow the request
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority_graph(self, synapse: MaximumCliqueOfLambdaGraph) -> float:
        return await self.priority(synapse)


if __name__ == "__main__":
    miner = None
    try:
        miner = Miner()
        with miner:
            bt.logging.info("Miner has started running.")
            while True:
                if miner.should_exit:
                    bt.logging.info("Miner is exiting.")
                    break
                time.sleep(1)
    finally:
        # Clean up thread pool executor
        if miner and hasattr(miner, 'executor'):
            miner.executor.shutdown(wait=True)
            bt.logging.info("Thread pool executor shut down.")
