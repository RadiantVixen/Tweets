     
def nextPermutation(nums: list[int]) -> None:

    r = len(nums) - 1
    l = len(nums) - 2
    i = 0

    if nums == sorted(nums, reverse=True):
        nums.sort()
        return

    while l >= 0:
        nums[l], nums[r] = nums[r], nums[l]
        if nums[l] > nums[r]:
            break
        l = l - 1
        r = r - 1
        if l < 0:
            l = len(nums) - 1
        
        if r < 0:
            r = len(nums) - 1
        
        i += 1
        if i > len(nums) + :
            break
        print(nums)



nums = [5,4,7,5,3,2]
nextPermutation(nums)
print(nums)



[5,5,2,3,4,7]

