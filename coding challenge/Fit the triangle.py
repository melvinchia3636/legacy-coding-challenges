def does_triangle_fit(brick, hole):
        return True if hole[0]>=brick[0] and hole[1]>=brick[1] and hole[2]>=brick[2] and hole[0]+hole[1]>hole[2] and hole[0]+hole[2]>hole[1] and hole[1]+hole[2]>hole[0] and brick[0]+brick[1]>brick[2] and brick[0]+brick[2]>brick[1] and brick[1]+brick[2]>brick[0] else False
