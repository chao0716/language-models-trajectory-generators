# INPUT: [INSERT AUDIO TEXT]
TASK_COMMAND_FROM_AUDIO = \
"""
The user has provided the following instruction in the form of audio, which has been converted to text:

"[INSERT AUDIO TEXT]"

Your task is to summarize the user's request and generate a specific command or task description that the robot should perform. Make sure to consider the intent behind the user's words and provide a clear, concise task command that the robot can execute.

Respond with the task command in sentences and output the task command in the following format:

<<command>>[Your Task Command Here]</command>>
"""
