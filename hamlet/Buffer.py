class Buffer:
    _instance = None
    configs = {}
    results = {}
    current_iteration = 0

    def __new__(buffer):
        if buffer._instance is None:
            buffer._instance = super(Buffer, buffer).__new__(buffer)
        return buffer._instance

    def reset(buffer):
        buffer.configs = {}
        buffer.results = {}
        buffer.current_iteration = 0

    def add_evaluation(buffer, config, result):
        buffer.configs[buffer.current_iteration] = config
        buffer.results[buffer.current_iteration] = result
        buffer.current_iteration += 1

    def get_evaluations(buffer):
        points_to_evaluate = list(buffer.configs.values())
        evaluated_rewards = [result["accuracy"] for result in buffer.results.values()]
        return points_to_evaluate, evaluated_rewards
