__kernel void pi_calculate(
   int term_num,
   __local float *local_result,
   __global float *global_result){
   
   // work item ind
   int item_id = get_global_id(0);   
   local_result[item_id] = 0;

   // pi calculation within a work item
   for(int i = 1; i < term_num*2; i = i+4){
      float tmp1 = (float)1./(item_id*term_num*2+i);      // find the term by index
      float tmp2 = (float)1./(item_id*term_num*2+i+2);
      local_result[item_id] += tmp1 - tmp2; //two terms substract as a basic cell
   } 

   // Barrier function
   barrier(CLK_LOCAL_MEM_FENCE);
   // send the local results to global
   global_result[item_id] = local_result[item_id];
   
   // print out results
   printf("Work item %d, corresponding value: %f \n", item_id, local_result[item_id]);
   
   return;
}
