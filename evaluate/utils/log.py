import logging
import colorlog

MDP_CREATION = 20
LEARNING = 21
EVALUATING = 22

logging.addLevelName(MDP_CREATION, 'MDP_CREATION')
logging.addLevelName(LEARNING, 'LEARNING')
logging.addLevelName(EVALUATING, 'EVALUATING')

formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s<%(process)d> %(asctime)s: %(message)s',
    log_colors={'MDP_CREATION': 'yellow',
                'LEARNING': 'green',
                'EVALUATING': 'blue'},
    datefmt="%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.Logger.mdp_creation = lambda self, msg: self.log(MDP_CREATION, msg)
logging.Logger.learning = lambda self, msg: self.log(LEARNING, msg)
logging.Logger.evaluating = lambda self, msg: self.log(EVALUATING, msg)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)

logger = logging.getLogger("HAL EVALUATION")
# logger2 = logging.getLogger('asdf3')
# # logger.addHandler(handler)
# # logger.setLevel('DEBUG')
# logger2.learning('a message using a custom level')
# logger.evaluating('a message using a custom level')