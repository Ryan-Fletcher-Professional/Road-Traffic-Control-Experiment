from unittest import TestCase

class TestAgentOrderingConsistent(TestCase):
    def __init__(self, methodName='runTest', param=None):
        super(TestAgentOrderingConsistent, self).__init__(methodName)
        self.agents = param
    def runTest(self):
        self.assertEquals(",1863241632,2330725114,243351999,243641585,243749571,30503246,30624898,32564122,89127267,89173763,89173808,cluster_1427494838_273472399,cluster_1757124350_1757124352,cluster_1863241547_1863241548_1976170214,cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190,gneJ143,gneJ207,gneJ208,gneJ210,gneJ255,gneJ257", self.agents, "Inconsistent Agent Ordering")