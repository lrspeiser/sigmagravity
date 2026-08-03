import { consumeQueueCanary, createQueueNodeHandler } from "../../lib/production-queue.mjs";

export default createQueueNodeHandler(async (message) => {
  await consumeQueueCanary(message);
});
