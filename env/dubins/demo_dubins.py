import cv2
from dubins_wrapper import DubinsWrapper

if __name__ == '__main__':
    env = DubinsWrapper(seed=0)
    init_state, goal_state = env.sample_random_init_goal_states(seed=42)
    obs, state = env.prepare(0, init_state)
    print('Initial full state:', state)
    print('Initial proprio (agent):', obs['proprio'])
    print('Initial h:', env.compute_h(state))
    cv2.imshow('step0', obs['visual']); cv2.waitKey(500)
    for t in range(1, 1010):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f'Step {t} full state:', info['state'])
        print('  proprio:', obs['proprio'], 'h:', info['h'])
        cv2.imshow('step', obs['visual'])
        if cv2.waitKey(200) & 0xFF == 27 or done:
            print('Done:', done)
            break
    cv2.destroyAllWindows()
