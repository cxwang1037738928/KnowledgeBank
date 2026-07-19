-- DropIndex
DROP INDEX "Collection_guestId_idx";

-- AlterTable
ALTER TABLE "Collection" DROP COLUMN "guestId",
ALTER COLUMN "userId" SET NOT NULL;

