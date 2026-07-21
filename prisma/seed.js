/**
 * seed.js — creates the admin account (admin@gmail.com / admin123) and the
 * demo account (demo@gmail.com / demo123) shown on the login screen.
 * Run: npm run db:seed (idempotent — upserts by email).
 */

import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

const ACCOUNTS = [
  { email: 'admin@gmail.com', password: 'admin123', isAdmin: true },
  { email: 'demo@gmail.com',  password: 'demo123',  isAdmin: false },
];

async function main() {
  for (const account of ACCOUNTS) {
    const password = await bcrypt.hash(account.password, 10);
    // Reset the password on every run so a rotated demo credential in this
    // file always matches what the login screen advertises.
    const user = await prisma.user.upsert({
      where:  { email: account.email },
      update: { password, isAdmin: account.isAdmin },
      create: { email: account.email, password, isAdmin: account.isAdmin },
    });
    console.log(`[seed] user ready (id=${user.id}, email=${user.email})`);
  }
}

main()
  .catch((err) => { console.error('[seed]', err); process.exit(1); })
  .finally(() => prisma.$disconnect());
