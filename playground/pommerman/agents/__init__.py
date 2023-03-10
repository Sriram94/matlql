'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .simple_agent_new_differentquality import SimpleAgentNew_differentquality
from .simple_agent_new_similarquality import SimpleAgentNew_similarquality
from .simple_agent_new_insufficient_similarquality import SimpleAgentNew_insufficient_similarquality
from .simple_agent_new_insufficient_differentquality import SimpleAgentNew_insufficient_differentquality
from .random_agent import RandomAgent
from .simple_agent import SimpleAgent
from .dqn import DQNAgent
from .dqn_differentagents import DQNAgent_differentagents
from .admiraldm_differentquality import admiraldm_differentquality 
from .admiraldm_similarquality import admiraldm_similarquality
from .admiraldm_insufficient_similarquality import admiraldm_insufficient_similarquality
from .admiraldm_insufficient_differentquality import admiraldm_insufficient_differentquality
from .admiraldmteamcomp import admiraldmteam 
from .dqfd import DQfDAgent
from .chat_differentquality import CHATAgent_differentquality
from .chat_similarquality import CHATAgent_similarquality
from .chat_insufficient_similarquality import CHATAgent_insufficient_similarquality
from .chat_insufficient_differentquality import CHATAgent_insufficient_differentquality
from .tlql_differentquality import twolevelqlAgent_differentquality
from .tlqlJA_differentquality import twolevelqlAgentJA_differentquality
from .tlqlAE_differentquality import twolevelqlAgentAE_differentquality
from .tlqlEM_differentquality import twolevelqlAgentEM_differentquality
from .tlql_similarquality import twolevelqlAgent_similarquality
from .tlqlJA_similarquality import twolevelqlAgentJA_similarquality
from .tlqlEM_similarquality import twolevelqlAgentEM_similarquality
from .tlqlAE_similarquality import twolevelqlAgentAE_similarquality
from .tlql_insufficient_similarquality import twolevelqlAgent_insufficient_similarquality
from .tlql_insufficient_differentquality import twolevelqlAgent_insufficient_differentquality
from .matlql_differentquality import matwolevelqlAgent_differentquality
from .matlql_similarquality import matwolevelqlAgent_similarquality
from .matlql_insufficient_similarquality import matwolevelqlAgent_insufficient_similarquality
from .matlql_insufficient_differentquality import matwolevelqlAgent_insufficient_differentquality
from .matlql_insufficient_differentqualityoppmod import matwolevelqlAgent_insufficient_differentqualityoppmod
from .matlql_insufficient_similarqualityoppmod import matwolevelqlAgent_insufficient_similarqualityoppmod
from .matlqlteamcomp import matwolevelqlteamAgent
from .matlac_differentquality import matwolevelacAgent_differentquality
from .matlac_similarquality import matwolevelacAgent_similarquality
from .matlac_insufficient_similarquality import matwolevelacAgent_insufficient_similarquality
from .matlac_insufficient_differentquality import matwolevelacAgent_insufficient_differentquality
from .matlacteamcomp import matwolevelacteamAgent
from .advisor1_differentquality import Advisor1_differentquality
from .advisor2_differentquality import Advisor2_differentquality
from .advisor3_differentquality import Advisor3_differentquality
from .advisor4_differentquality import Advisor4_differentquality
from .advisor1_insufficient_similarquality import Advisor1_insufficient_similarquality
from .advisor1_insufficient_differentquality import Advisor1_insufficient_differentquality
from .advisor1_similarquality import Advisor1_similarquality
from .advisor2_similarquality import Advisor2_similarquality
from .advisor2_insufficient_similarquality import Advisor2_insufficient_similarquality
from .advisor2_insufficient_differentquality import Advisor2_insufficient_differentquality
from .advisor3_similarquality import Advisor3_similarquality
from .advisor3_insufficient_similarquality import Advisor3_insufficient_similarquality
from .advisor3_insufficient_differentquality import Advisor3_insufficient_differentquality
from .advisor4_similarquality import Advisor4_similarquality
from .advisor4_insufficient_similarquality import Advisor4_insufficient_similarquality
from .advisor4_insufficient_differentquality import Advisor4_insufficient_differentquality

